import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
from alpaca_trade_api.stream import Stream

# Load environment variables from a .env file
load_dotenv()

# Retrieve Alpaca API credentials from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Alpaca news stream WebSocket URL
NEWS_STREAM = "wss://stream.data.alpaca.markets/v1beta1/news"

# Sentiment analysis API endpoint
SENTIMENT_API_URL = "http://sentim-Publi-poYLmNKOtgZE-567569739.us-east-1.elb.amazonaws.com/predict"

# Initialize Alpaca news stream with API credentials
stream = Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, raw_data=True)

# Function to get sentiment analysis for a given text
async def get_sentiment(text: str):
    async with aiohttp.ClientSession() as session:
        # Send a POST request to the sentiment analysis API
        async with session.post(SENTIMENT_API_URL, json={"text": text}) as resp:
            if resp.status == 200:
                # Return the JSON response if the request is successful
                return await resp.json()
            else:
                # Return an error response if the request fails
                return {"label": "ERROR", "score": 0.0}

# Callback function to handle incoming news data
async def news_handler(news):
    # Extract headline and creation timestamp from the news data
    headline = news.get('headline')
    created_at = news.get('created_at')    

    # Perform sentiment analysis on the headline
    sentiment = await get_sentiment(headline)

    # Print the news headline and its sentiment analysis result
    print(f"[{created_at}] {headline}")
    print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']})\n")

# Subscribe to all news events and set the handler function
stream.subscribe_news(news_handler, "*")

# Start the Alpaca news stream and run it indefinitely
asyncio.run(stream._run_forever())