from src.util import BaseAgent


class BasicAgent(BaseAgent):
    def __init__(self):
        print("BasicAgent initialized.")

    def run(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is the BasicAgent's default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer
