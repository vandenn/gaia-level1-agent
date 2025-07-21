import sys

from src.agent import GaiaAgent
from src.settings import settings
from src.ui import initialize

if __name__ == "__main__":
    if "--test" in sys.argv:
        # question = "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation."
        question = "Review the chess position provided in the image. Where is the black knight, black rook, and white rook?"
        file_name = "cca530fc-4052-43b2-b130-b30968d8aa44.png"
        api_url = settings.default_api_url
        file_url = f"{api_url}/files/cca530fc-4052-43b2-b130-b30968d8aa44"
        agent = GaiaAgent()
        print(f"Response: {agent.run(question, file_name, file_url)}")
    else:
        initialize(GaiaAgent())
