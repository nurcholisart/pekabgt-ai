from pydantic import BaseModel


class Article(BaseModel):
    title: str
    link: str
    content: str
