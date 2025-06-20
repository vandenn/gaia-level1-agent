from collections import deque
import time
import random
from typing import Any

from smolagents import DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, WikipediaSearchTool, ToolCallingAgent
from smolagents.agents import FinalAnswerStep
from smolagents.tools import Tool

from src.settings import settings
from src.util import BaseAgent


class GaiaAgent(BaseAgent):
    def __init__(self):
        self.model = LiteLLMModel(
            model_id="anthropic/claude-3-5-haiku-20241022",
            api_key=settings.anthropic_api_key
        )
        self.agent = ToolCallingAgent(
            tools=[
                DuckDuckGoSearchTool(),
                FinalAnswerTool(),
                WikipediaSearchTool(),
                VisitWebpageTool(),
            ],
            planning_interval=3,
            max_steps=5,
            model=self.model,
        )
        self.token_rate_limiter = InputTokenRateLimiter()
        self.expected_tokens_per_step = 10000
        self.max_retries = 5
        self.base_delay = 5

    def run(self, question: str) -> Any:
        final_answer = None
        retry_count = 0

        while True:
            try:
                for step in self.agent.run(question, stream=True):
                    self.token_rate_limiter.maybe_wait(self.expected_tokens_per_step)
                    token_usage_info = getattr(step, "token_usage", None)
                    tokens_used = 0
                    if tokens_used:
                        tokens_used = token_usage_info.input_tokens
                    self.token_rate_limiter.add_tokens(tokens_used)
                    if isinstance(step, FinalAnswerStep):
                        final_answer = step.output
                break
            except Exception as e:
                if "overloaded" in str(e).lower() or "rate limit" in str(e).lower() or "529" in str(e):
                    if retry_count >= self.max_retries:
                        print(f"Max retries reached. Error: {e}")
                    delay = self.base_delay * (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Anthropic server error due to overload or rate limit. Retrying in {delay:.1f} seconds..")
                    time.sleep(delay)
                    retry_count += 1
                else:
                    print(f"Error occurred: {e}")

        return final_answer


SECONDS_IN_MINUTE = 60
class InputTokenRateLimiter:
    _instance = None

    def __new__(cls): # Singleton
        if cls._instance is None:
            cls._instance = super(InputTokenRateLimiter, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_tpm=50000):
        self.max_tpm = max_tpm
        self.token_window = deque() # stores (timestamp, tokens_used)
        
    def _update_queue(self, time_now):
        while self.token_window and time_now - self.token_window[0][0] > SECONDS_IN_MINUTE:
            self.token_window.popleft()

    def tokens_used_last_minute(self):
        now = time.time()
        self._update_queue(now)
        return sum(tokens for _, tokens in self.token_window)

    def maybe_wait(self, tokens_expected_to_use):
        ctr = 0
        while self.tokens_used_last_minute() + tokens_expected_to_use > self.max_tpm:
            if ctr % 10 == 0:
                print(f"Tokens: {self.token_window}")
                print("slept 5 seconds")
            time.sleep(0.5)
            now = time.time()
            self._update_queue(now)
            ctr += 1

    def add_tokens(self, tokens):
        now = time.time()
        self.token_window.append((now, tokens))
        self._update_queue(now)


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides the exact, final answer to the given question."
    inputs = {
        "answer": {"type": "string", "description": "The exact, final answerf to the question."}
    }
    output_type = "string"

    def forward(self, answer: str) -> str:
        return answer


if __name__ == "__main__":
    agent = GaiaAgent()
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of English wikipedia."
    print(f"Response: {agent.run(question)}")
