from pydantic import BaseModel, Field


class VisionAnalysisResponse(BaseModel):
    diagnosis: str = Field(description="Description textuelle du contenu visuel générée par IA")
    label: str = Field(description='Catégorie générale ou statut (ex: analyse_visuelle_terminee)')
