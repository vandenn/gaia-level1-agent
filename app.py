from src.agent import GaiaAgent
from src.ui import initialize
import sys

if __name__ == "__main__":
    if "--test" in sys.argv:
        question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of English wikipedia."
        agent = GaiaAgent()
        print(f"Response: {agent.run(question)}")
    else:
        initialize(GaiaAgent())
