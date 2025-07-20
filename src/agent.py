import random
import time
from typing import Any

from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    PythonInterpreterTool,
    ToolCallingAgent,
    VisitWebpageTool,
)
from smolagents.agents import FinalAnswerStep

from src.settings import settings
from src.tools import DownloadAndParseFileTool, FinalAnswerTool
from src.utils import BaseAgent, InputTokenRateLimiter


class GaiaAgent(BaseAgent):
    def __init__(self):
        self.model = LiteLLMModel(
            model_id=settings.llm_model_id, api_key=settings.llm_api_key
        )
        self.agent = ToolCallingAgent(
            tools=[
                DuckDuckGoSearchTool(max_results=3),
                VisitWebpageTool(max_output_length=20000),
                DownloadAndParseFileTool(),
                PythonInterpreterTool(),
                FinalAnswerTool(),
                # TODO: Image interpretation, MP3 interpretation
            ],
            max_steps=10,
            planning_interval=5,
            model=self.model,
        )
        self.token_rate_limiter = InputTokenRateLimiter()
        self.expected_tokens_per_step = 10000
        self.max_retries = 3
        self.base_delay = 5

    def run(self, question: str, file_name: str = "", file_url: str = "") -> Any:
        final_answer = None
        retry_count = 0

        input = f"""
            Answer the following QUESTION as concisely as possible. A necessary FILE may be provided as part of the context of the QUESTION.
            Make the shortest possible execution plan to answer this QUESTION.

            QUESTION: {question}
            FILE NAME: {file_name if file_name else "N/A"}
            FILE URL: {file_url if file_url else "N/A"}
        """

        while True:
            try:
                for step in self.agent.run(input, stream=True):
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
                if (
                    "overloaded" in str(e).lower()
                    or "rate limit" in str(e).lower()
                    or "529" in str(e)
                ):
                    if retry_count >= self.max_retries:
                        print(f"Max retries reached. Error: {e}")
                        break
                    delay = self.base_delay * (2**retry_count) + random.uniform(0, 1)
                    print(
                        f"Anthropic server error due to overload or rate limit. Retrying in {delay:.1f} seconds.."
                    )
                    print(f"The error was: {e}")
                    time.sleep(delay)
                    retry_count += 1
                else:
                    print(f"Error occurred: {e}")
                    break

        return final_answer


if __name__ == "__main__":
    agent = GaiaAgent()
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of English wikipedia."
    print(f"Response: {agent.run(question)}")
