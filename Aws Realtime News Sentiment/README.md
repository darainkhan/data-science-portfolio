# 📰 Real-Time News Sentiment Analysis API

A production-ready ML inference service that scores financial news headlines for sentiment in real-time. Streams live market news via Alpaca's WebSocket API, classifies each headline using a transformer-based NLP model, and serves predictions through a FastAPI REST endpoint — deployed on AWS behind a load balancer.

---

## Architecture

```
Alpaca News WebSocket ──→ Stream Consumer ──→ Sentiment API ──→ JSON Response
     (live headlines)        (stream/)           (app/)         {label, score}
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API Framework | FastAPI + Uvicorn | Serves predictions via REST endpoint |
| NLP Model | DistilBERT (HuggingFace Transformers) | Classifies text as POSITIVE / NEGATIVE |
| Streaming | Alpaca News WebSocket | Ingests real-time financial headlines |
| Validation | Pydantic | Request/response schema enforcement |
| Containerization | Docker | Reproducible builds for deployment |
| Deployment | AWS ECS + ELB | Scalable cloud inference |
| Inference | PyTorch (MPS/GPU-accelerated) | Low-latency model execution |

---

## How It Works

1. **Alpaca streams a live news headline** (e.g., "Tesla beats Q3 earnings expectations")
2. **The stream consumer** (`stream/news.py`) receives it via WebSocket
3. **It sends the headline** to the sentiment API (`POST /predict`)
4. **The API runs it through DistilBERT** and returns a label + confidence score
5. **Result:** `{"label": "POSITIVE", "score": 0.9732}`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — confirms API is live |
| `POST` | `/predict` | Accepts `{"text": "..."}`, returns sentiment label and score |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazon stock hits all-time high after strong earnings"}'
```

### Example Response

```json
{"label": "POSITIVE", "score": 0.9961}
```

---

## Project Structure

```
aws-realtime-news-sentiment/
├── app/
│   ├── main.py          # FastAPI application with endpoints
│   ├── sentiment.py     # Model loading and prediction logic
│   └── schemas.py       # Pydantic input/output schemas
├── stream/
│   └── news.py          # Real-time Alpaca news consumer
├── Dockerfile           # Container build for AWS deployment
├── pyproject.toml       # Dependencies and project metadata
└── .env                 # API credentials (not committed)
```

---

## Key Technical Decisions

- **Pre-trained model over custom training** — DistilBERT fine-tuned on SST-2 provides strong general sentiment classification without the cost of collecting and labeling financial text data. Suitable for headline-level analysis where context is short.
- **FastAPI over Flask** — Async support, automatic OpenAPI docs, built-in request validation via Pydantic, and better performance under concurrent load.
- **Separation of concerns** — Model logic (`sentiment.py`), API routing (`main.py`), and schemas (`schemas.py`) are decoupled for testability and maintainability.
- **Docker for deployment** — Ensures consistent behavior between local development and AWS, eliminates "works on my machine" issues.

---

## Running Locally

```bash
# Install dependencies
uv sync

# Start the API server
uv run uvicorn app.main:app --reload --port=8000

# Test it (in a separate terminal)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Markets rally on strong jobs report"}'
```

---

## Deployment

The service is containerized and deployed to AWS:

```bash
# Build the Docker image
docker build -t sentiment-api .

# Run locally via Docker
docker run -p 8000:8000 sentiment-api
```

Production deployment uses AWS ECS with an Elastic Load Balancer for scalability and high availability.

---

## Tech Stack

Python · FastAPI · PyTorch · HuggingFace Transformers · Pydantic · Alpaca API · WebSockets · Docker · AWS ECS · AWS ELB

---

## Skills Demonstrated

| Skill | How |
|-------|-----|
| ML Model Serving | Wrapping a transformer model in a production REST API |
| Real-Time Data Streaming | WebSocket consumer processing live financial news |
| API Design | RESTful endpoints with schema validation and error handling |
| Containerization | Dockerfile with multi-stage-friendly structure for cloud deployment |
| Cloud Deployment | AWS ECS + ELB for scalable inference |
| NLP / Transformers | Using pre-trained HuggingFace models for text classification |
