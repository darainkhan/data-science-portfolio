from pydantic import BaseModel

# Schema for input text data
class TextInput(BaseModel):
    text: str  # The input text for sentiment analysis

# Schema for sentiment analysis output
class SentimentOutput(BaseModel):
    label: str  # The predicted sentiment label (e.g., positive, negative, neutral)
    score: float  # The confidence score of the sentiment prediction