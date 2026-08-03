from enum import Enum

from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    REFUNDABLE = 'Remboursable'
    NEEDS_REVIEW = 'À vérifier'
    REJECTED = 'Refusé'

class TicketResponse(BaseModel):
    transcription: str | None = Field(default=None, description="La transcription de l'audio s'il y en avait un")
    image_diagnosis: str | None = Field(default=None, description="Le diagnostic de l'image s'il y en avait une")
    applied_rule: str = Field(description="La règle de gestion trouvée dans la base documentaire")
    status: TicketStatus = Field(description="Le statut final du ticket calculé par le Gateway")
