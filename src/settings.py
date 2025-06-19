from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    hf_token: str = Field(default=...)
    default_api_url: str = Field(default=...)


settings = Settings()  # from settings import settings
