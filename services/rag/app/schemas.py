from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    query_text: str = Field(description='Le texte fusionné par le Gateway')
    top_k: int = Field(default=1, description='Nombre de règles à remonter')

class RAGQueryResponse(BaseModel):
    rule_title: str
    rule_content: str
    similarity_score: float = Field(description='Score de similarité entre la requête et la règle')
