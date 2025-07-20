import mimetypes
import random
import time
from io import BytesIO
from typing import Any, TypedDict

import pandas as pd
import requests
from PIL import Image
from smolagents import (
    DuckDuckGoSearchTool,
    LiteLLMModel,
    PythonInterpreterTool,
    ToolCallingAgent,
    VisitWebpageTool,
)
from smolagents.agents import FinalAnswerStep

from src.settings import settings
from src.tools import FinalAnswerTool
from src.utils import BaseAgent, InputTokenRateLimiter


class ParsedFile(TypedDict):
    text: str
    image: Image


class GaiaAgent(BaseAgent):
    def __init__(self):
        self.model = LiteLLMModel(
            model_id=settings.llm_model_id, api_key=settings.llm_api_key
        )
        self.agent = ToolCallingAgent(
            tools=[
                DuckDuckGoSearchTool(max_results=3),
                VisitWebpageTool(max_output_length=20000),
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

        parsed_file = self._parse_file(file_name, file_url)
        input = f"""
            Answer the following QUESTION as concisely as possible.
            If available, a FILE NAME and the actual FILE is attached for your reference.
            Make the shortest possible execution plan to answer this QUESTION.

            QUESTION: {question}
            FILE NAME: {file_name if file_name else "N/A"}
        """
        if parsed_file["text"]:
            input = input + f"\nFILE CONTENT: {parsed_file['text']}"
        input_images = None
        if parsed_file["image"]:
            input_images = [parsed_file["image"]]

        while True:
            try:
                for step in self.agent.run(input, images=input_images, stream=True):
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

    def _parse_file(self, file_name: str, file_url: str) -> ParsedFile:
        result = ParsedFile(text=None, image=None)
        if not file_name or not file_url:
            return result

        try:
            response = requests.get(file_url)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to download file: {e}")
            return result

        # Try to handle the 'no file' JSON case
        try:
            file_data = response.json()
            if (
                "detail" in file_data
                and "No file path associated" in file_data["detail"]
            ):
                print(f"No file found for {file_name} at {file_url}")
                return result
        except Exception:
            pass  # Not JSON, so it's probably the file content

        file_type, _ = mimetypes.guess_type(file_name)
        if file_type and file_type.startswith("text"):
            try:
                result["text"] = response.content.decode("utf-8")
                return result
            except Exception:
                return "Failed to decode text file as utf-8."
        elif file_name.endswith(".py"):
            try:
                result["text"] = response.content.decode("utf-8")
                return result
            except Exception:
                return "Failed to decode Python file as utf-8."
        elif file_name.endswith(".xlsx"):
            try:
                df = pd.read_excel(BytesIO(response.content))
                result["text"] = df.to_string()
                return result
            except Exception as e:
                return f"Failed to parse Excel file: {e}"
        elif file_type and file_type.startswith("image"):
            try:
                image = Image.open(BytesIO(response.content))
                result["image"] = image
                return result
            except Exception as e:
                return f"Failed to decode image file: {e}"
        else:
            print(
                f"[{file_name} is a binary file of type {file_type or 'unknown'} and cannot be parsed as text.]"
            )
            return result


if __name__ == "__main__":
    agent = GaiaAgent()
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of English wikipedia."
    print(f"Response: {agent.run(question)}")
