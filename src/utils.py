from abc import ABC, abstractmethod
from typing import Any

from collections import deque
import time
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    def run(self, question: str, file_name: str = "", file_content: str = "") -> Any:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is the BaseAgent's default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer


SECONDS_IN_MINUTE = 60
class InputTokenRateLimiter:
    _instance = None

    def __new__(cls): # Singleton
        if cls._instance is None:
            cls._instance = super(InputTokenRateLimiter, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_tpm=50000):
        self.max_tpm = max_tpm
        if not hasattr(self, "_initialized"):
            self.token_window = deque() # stores (timestamp, tokens_used)
            self._initialized = True
        
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

