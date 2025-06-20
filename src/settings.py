from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    hf_token: str = Field(default=...)
    llm_model_id: str = Field(default="anthropic/claude-3-5-haiku-20241022")
    llm_api_key: str = Field(default=...)
    default_api_url: str = Field(default="https://agents-course-unit4-scoring.hf.space")


settings = Settings()  # from settings import settings
