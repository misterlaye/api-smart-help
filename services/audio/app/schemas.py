from pydantic import BaseModel, Field


class AudioTranscriptionResponse(BaseModel):
    transcription: str = Field(description="Texte brut transcrit à partir de l'audio")
