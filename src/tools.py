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
