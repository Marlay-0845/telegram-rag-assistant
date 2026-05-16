from pydantic import BaseModel


class RewriteQuestion(BaseModel):
    question: str
    section: str