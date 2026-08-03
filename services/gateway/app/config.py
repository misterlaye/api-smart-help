from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    audio_service_url: str = 'http://audio:8000'
    vision_service_url: str = 'http://vision:8000'
    rag_service_url: str = 'http://rag:8000'

    class Config:
        env_file = '.env'


settings = Settings()
