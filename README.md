# FastAPI Chatbot RAG Microservice

A production-ready FastAPI microservice providing real-time Retrieval-Augmented Generation (RAG) chatbot interactions powered by **Qdrant Cloud**, **Sentence Transformers**, and Anthropic's **Claude API** (`claude-sonnet-5`).

This microservice retrieves context from sacred Hindu scripture texts (Bhagavad Gita, Mahabharata, etc.) indexed in Qdrant Cloud to answer user questions with accurate citations. It runs alongside an existing **Node.js + MongoDB + JWT** main backend without requiring any modifications to the Node codebase.

---

## 🏗 System Architecture

```
                      ┌────────────────────────┐
                      │    Client (Frontend)   │
                      └───────────┬────────────┘
                                  │
                          Nginx Reverse Proxy
                         /                   \
           (All other routes)           (Route /chat/*)
                        /                     \
       ┌───────────────────────┐       ┌───────────────────────┐
       │   Node.js Main App    │       │  FastAPI Chatbot App  │
       │ (Auth / Business Log) │       │   (Port 8000)         │
       └───────────┬───────────┘       └───────────┬───────────┘
                   │                               │
        Shared JWT Secret              Verifies JWT with Shared Secret
                   │                               │
       ┌───────────▼───────────┐       ┌───────────▼───────────┐
       │   MongoDB Main DB     │       │   MongoDB Chatbot DB  │
       └───────────────────────┘       │   & Async Redis       │
                                       └───────────┬───────────┘
                                                   │
                                     ┌─────────────▼─────────────┐
                                     │    Qdrant Cloud Cluster   │
                                     │  (Vector Scripture Index) │
                                     └───────────────────────────┘
```

- **Authentication**: Verifies JWT tokens issued by the Node.js backend using a shared `JWT_SECRET`.
- **Fast Storage (Redis)**: Holds rolling context windows (last 10 messages, 1-hour TTL) for ultra-fast prompt assembly.
- **Persistent Storage (MongoDB)**: Stores permanent conversation histories (`conversations` and `messages` collections).
- **Vector Search (Qdrant Cloud)**: Cloud-hosted vector database storing embedded scripture text chunks (`gita_collection`, `mahabharata_collection`).
- **Streaming (SSE)**: Streams Claude RAG token responses in real-time using Server-Sent Events (`text/event-stream`).

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager
- Running MongoDB instance (`mongodb://localhost:27017`)
- Running Redis instance (`redis://localhost:6379`)
- Qdrant Cloud cluster account ([cloud.qdrant.io](https://cloud.qdrant.io/))

### 2. Install Dependencies
Run `uv sync` to install all dependencies from `pyproject.toml` and generate lockfile:
```bash
uv sync
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
JWT_SECRET=your_shared_jwt_secret_key_here
JWT_ALGORITHM=HS256
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=chatbot_db
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Qdrant Cloud Cluster Configuration
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_cloud_api_key_here

# Embedding & Vector RAG Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
AVAILABLE_COLLECTIONS=gita_collection,mahabharata_collection
```

---

## 🔮 Setting up Qdrant Cloud & Ingesting Documents

### 1. Create a Free Cluster on Qdrant Cloud
1. Sign up or log in at [cloud.qdrant.io](https://cloud.qdrant.io/).
2. Create a new free-tier cluster.
3. In the Cluster Dashboard, copy your **Cluster URL** (e.g. `https://xxxx.us-east-1-0.aws.cloud.qdrant.io:6333`) and generate an **API Key**.
4. Paste these values into your `.env` file under `QDRANT_URL` and `QDRANT_API_KEY`.

### 2. Ingesting Text Files (Offline Ingestion Script)
Document ingestion is a manual/offline process run via CLI. It can be run from any machine with network access to Qdrant Cloud without needing a local Qdrant instance.

1. Place your text files inside the `data/` directory (e.g., `data/gita.txt`).
2. Run the ingestion command:

```bash
uv run python -m app.scripts.ingest \
  --file data/gita.txt \
  --source "Bhagavad Gita" \
  --collection gita_collection \
  --chunk-size 500 \
  --chunk-overlap 50 \
  --batch-size 100
```

#### What the ingestion script does:
- Reads the raw text file.
- Splits text into chunks using `RecursiveCharacterTextSplitter`.
- Embeds chunks using `SentenceTransformer("all-MiniLM-L6-v2")` (384-dim).
- Automatically creates the collection on Qdrant Cloud if it doesn't exist.
- Upserts points in batches (100 at a time) with payload `{text, source, chunk_index}`.

### 3. Inspecting Data in Qdrant Cloud Console
Log in to [cloud.qdrant.io](https://cloud.qdrant.io/) and select **Collections** to visually inspect your indexed scripture collections, point counts, and metadata payloads.

---

## 🏃 Running the Server

Start the application using `uv`:
```bash
uv run uvicorn app.main:app --reload --port 8000
```

The server automatically pre-loads the embedding model and validates Qdrant Cloud connectivity during startup.

---

## 🔑 Generating a Test JWT Token

```bash
uv run python -c "import jwt, datetime; print(jwt.encode({'sub': 'user_123456789', 'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)}, 'your_shared_jwt_secret_key_here', algorithm='HS256'))"
```

---

## 🧪 Example API Calls

### 1. List Available Scripture Collections (`GET /chat/collections`)
```bash
curl -X GET http://localhost:8000/chat/collections
```
**Response:**
```json
{"collections": ["gita_collection", "mahabharata_collection"]}
```

### 2. Stream RAG Chat Response (`POST /chat/stream`)
```bash
TOKEN="YOUR_GENERATED_JWT_TOKEN"

curl -N -X POST http://localhost:8000/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does the Bhagavad Gita teach about duty and karma?",
    "collections": ["gita_collection"]
  }'
```

**Output (SSE Stream):**
```
data: {"conversation_id": "8f03c004-9467-4e78-bc4a-bfef582f3bb4"}

data: {"text": "According to the Bhagavad Gita (Source: Bhagavad Gita)..."}

data: [DONE]
```

---

## 🌐 Nginx Deployment & SSE Proxy Configuration

Set **`proxy_buffering off;`** in Nginx for real-time SSE streaming:

```nginx
location /chat/ {
    proxy_pass http://127.0.0.1:8000/chat/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Disable buffering for SSE real-time streaming
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 24h;
    chunked_transfer_encoding on;
}
```
