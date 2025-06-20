from collections import deque
import time
import random
import mimetypes
from typing import Any
import requests
import pandas as pd
from io import BytesIO

from smolagents import DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, ToolCallingAgent, PythonInterpreterTool
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
                VisitWebpageTool(),
                DownloadAndParseFileTool(),
                PythonInterpreterTool(),
                # TODO: Image interpretation, MP3 interpretation
            ],
            planning_interval=3,
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
            Answer the following QUESTION as concisely as possible.
            Make the shortest possible execution plan to answer this QUESTION.

            QUESTION: {question}
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
                if "overloaded" in str(e).lower() or "rate limit" in str(e).lower() or "529" in str(e):
                    if retry_count >= self.max_retries:
                        print(f"Max retries reached. Error: {e}")
                        break
                    delay = self.base_delay * (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Anthropic server error due to overload or rate limit. Retrying in {delay:.1f} seconds..")
                    print(f"The error was: {e}")
                    time.sleep(delay)
                    retry_count += 1
                else:
                    print(f"Error occurred: {e}")
                    break

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
        "question": {"type": "string", "description": "The original question being asked."},
        "answer": {"type": "string", "description": "The answer to the question."}
    }
    output_type = "string"

    def __init__(self):
        self.model = LiteLLMModel(
            model_id="anthropic/claude-3-5-haiku-20241022",
            api_key=settings.anthropic_api_key,
            temperature=0.1,
            max_tokens=20
        )
        self.token_rate_limiter = InputTokenRateLimiter()
        self.expected_tokens_per_step = 10000
        self.is_initialized = True

    def forward(self, question: str, answer: str) -> str:
        self.token_rate_limiter.maybe_wait(self.expected_tokens_per_step)
        response = self.model.generate(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                                Rewrite the following ANSWER to be concise and use as few tokens as possible to answer the QUESTION directly.
                                If the answer is an error, return 'N/A' instead.

                                QUESTION: {question}
                                ANSWER: {answer}
                            """
                        }
                    ]
                }
            ]
        )
        token_usage_info = getattr(response, "token_usage", None)
        tokens_used = 0
        if tokens_used:
            tokens_used = token_usage_info.input_tokens
        self.token_rate_limiter.add_tokens(tokens_used)
        return response.content


class DownloadAndParseFileTool(Tool):
    name = "download_and_parse_file"
    description = "Downloads a file from a given URL and parses it based on the file name. Returns the file content as text if possible, or nothing if image, etc."
    inputs = {
        "file_name": {"type": "string", "description": "The name of the file (used to determine type)."},
        "file_url": {"type": "string", "description": "The URL of the file to download."},
    }
    output_type = "string"

    def __init__(self):
        self.is_initialized = True

    def forward(self, file_name: str, file_url: str) -> str:
        try:
            response = requests.get(file_url)
            response.raise_for_status()
        except Exception as e:
            return f"Failed to download file: {e}"

        # Try to handle the 'no file' JSON case
        try:
            file_data = response.json()
            if "detail" in file_data and "No file path associated" in file_data["detail"]:
                return f"No file found for {file_name} at {file_url}"
        except Exception:
            pass  # Not JSON, so it's probably the file content

        file_type, _ = mimetypes.guess_type(file_name)
        if file_type and file_type.startswith("text"):
            try:
                return response.content.decode("utf-8")
            except Exception:
                return "Failed to decode text file as utf-8."
        elif file_name.endswith(".py"):
            try:
                return response.content.decode("utf-8")
            except Exception:
                return "Failed to decode Python file as utf-8."
        elif file_name.endswith(".xlsx"):
            try:
                df = pd.read_excel(BytesIO(response.content))
                return df.to_string()
            except Exception as e:
                return f"Failed to parse Excel file: {e}"
        else:
            return f"[{file_name} is a binary file of type {file_type or 'unknown'} and cannot be parsed as text.]"


if __name__ == "__main__":
    agent = GaiaAgent()
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of English wikipedia."
    print(f"Response: {agent.run(question)}")
