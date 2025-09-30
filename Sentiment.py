from pydantic import BaseModel
from typing import Literal

class Sentiment(BaseModel):
    sentiment: Literal["positive", "neutral", "negative", "ERROR"]