from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    def run(self, question: str, file_name: str = "", file_content: str = "") -> Any:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is the BaseAgent's default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

