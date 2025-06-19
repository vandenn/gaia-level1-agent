from src.util import BaseAgent

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, WikipediaSearchTool
from smolagents.tools import Tool

from src.settings import settings


class GaiaAgent(BaseAgent):
    def __init__(self):
        self.model = LiteLLMModel(
            model_id="anthropic/claude-3-haiku-20240307",
            api_key=settings.anthropic_api_key
        )
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                FinalAnswerTool(),
                WikipediaSearchTool(),
                VisitWebpageTool(),
            ],
            planning_interval=3,
            model=self.model,
        )

    def run(self, question: str) -> str:
        return self.agent.run(question)


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
