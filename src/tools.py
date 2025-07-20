import mimetypes
from io import BytesIO

import pandas as pd
import requests
from smolagents import LiteLLMModel
from smolagents.tools import Tool

from src.settings import settings
from src.utils import InputTokenRateLimiter


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides the exact, final answer to the given question."
    inputs = {
        "question": {
            "type": "string",
            "description": "The original question being asked.",
        },
        "answer": {"type": "string", "description": "The answer to the question."},
    }
    output_type = "string"

    def __init__(self):
        self.model = LiteLLMModel(
            model_id=settings.llm_model_id,
            api_key=settings.llm_api_key,
            temperature=0.1,
            max_tokens=20,
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
                                If there's ambiguity in the ANSWER, make a clear cut decision to give a concise result.
                                Final result should not be in sentence format.
                                If the answer is an error, return 'N/A' instead.

                                QUESTION: {question}
                                ANSWER: {answer}
                            """,
                        }
                    ],
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
        "file_name": {
            "type": "string",
            "description": "The name of the file (used to determine type).",
        },
        "file_url": {
            "type": "string",
            "description": "The URL of the file to download.",
        },
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
            if (
                "detail" in file_data
                and "No file path associated" in file_data["detail"]
            ):
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
