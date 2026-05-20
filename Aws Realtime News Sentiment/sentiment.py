from transformers import pipeline

# Load sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"  # Or the latest revision you want to use
)

texts = [
    "I love this product!",
    "This is the worst experience I've ever had.",
    "I'm feeling okay about the results.",
    "Absolutely fantastic service!",
    "Not what I expected, but it's alright."
]

# Run sentiment analysis on each text and output the results
for text in texts:
    result = sentiment_pipeline(text)
    print(f"Text: {text}\nSentiment: {result}\n")