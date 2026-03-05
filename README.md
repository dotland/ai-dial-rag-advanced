# RAG (Retrieval-Augmented Generation) Implementation

A Python implementation task to build a complete RAG system for microwave manual assistance using PostgreSQL with pgvector extension and DIAL API

## 🎓 Learning Goals

By completing this task, you will learn:

- How to implement the complete RAG pipeline: **Retrieval**, **Augmentation**, and **Generation**
- How to work with vector embeddings and similarity search using PostgreSQL with pgvector
- How to process and chunk text documents for vector storage
- How to perform semantic search with cosine and Euclidean distance metrics
- Understanding RAG architecture without high-level frameworks

## 📋 Requirements

- Python 3.11+
- Docker and Docker Compose
- pip
- API key for DIAL service

## 🔧 Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**
    - Ensure that you are connected to the EPAM VPN
    - Get the DIAL API key here: <https://support.epam.com/ess?id=sc_cat_item&table=sc_cat_item&sys_id=910603f1c3789e907509583bb001310c>
    - Update `task/_constants.py` with your API credentials

3. **Start PostgreSQL with pgvector:**

   ```bash
   docker-compose up -d
   ```

   This will start PostgreSQL on port 5433 with the pgvector extension enabled.

4. **Project structure:**

   ```txt
   task/
   ├── _constants.py                    # 🚧 TODO - Add DIAL API KEY
   ├── app.py                           # 🚧 TODO - Main RAG application
   ├── embeddings/
   │   ├── embeddings_client.py         # 🚧 TODO - DIAL embeddings client
   │   ├── text_processor.py            # 🚧 TODO - Text processing and vector operations
   │   └── microwave_manual.txt         # ✅ Knowledge base document
   ├── chat/
   │   └── chat_completion_client.py    # ✅ DIAL chat completion client
   ├── models/
   │   ├── conversation.py              # ✅ Conversation management
   │   ├── message.py                   # ✅ Message structure
   │   └── role.py                      # ✅ Role definitions
   └── utils/
       └── text.py                      # ✅ Text chunking utilities
   docker-compose.yml                   # ✅ PostgreSQL with pgvector
   init-scripts/init.sql                # ✅ Database initialization
   ```

## 🖌️ Application Architecture:

The RAG system follows a three-step process:

1. **🔍 Retrieval**: Find relevant chunks from the microwave manual using vector similarity search
2. **🔗 Augmentation**: Combine retrieved context with user question in a structured prompt
3. **🤖 Generation**: Use LLM to generate accurate answer based on the provided context

## 📝 Your Tasks

### If the task in the main branch is hard for you, then switch to the `with-detailed-description` branch

Complete the implementation by filling in all the TODO sections across these files:

### 🔌 **Step 1: Embeddings Client (`embeddings_client.py`)**

- Complete the `get_embeddings()` method to call DIAL embeddings API
- Parse the response and extract embeddings data
- Handle the request/response format according to DIAL API specification

### 📊 **Step 2: Text Processing (`text_processor.py`)**

- Implement `process_text_file()` to load, chunk, and store document embeddings
- Complete `_truncate_table()` for database cleanup
- Implement `_save_chunk()` to store text chunks with embeddings in PostgreSQL
- Complete `search()` method for semantic similarity search using pgvector

### 🚀 **Step 3: Main Application (`app.py`)**

- Initialize DIAL clients with proper model deployments
- Implement document processing workflow
- Complete the RAG pipeline: Retrieval → Augmentation → Generation
- Handle user interaction and conversation management

## 🔧 Implementation Details

### Database Schema

The PostgreSQL database uses the pgvector extension with this schema:

```sql
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(64),
    text TEXT NOT NULL,
    embedding VECTOR(1536)
);
```

### Similarity Search

The system supports two distance metrics:

- **Cosine Distance** (`<=>` operator): Measures angle between vectors
- **Euclidean Distance** (`<->` operator): Measures straight-line distance

### Configuration Parameters

Experiment with these parameters for optimal performance:

- `chunk_size`: Size of text chunks (default: 150, recommended: 300)
- `overlap`: Character overlap between chunks (default: 40)
- `top_k`: Number of relevant chunks to retrieve (default: 5)
- `min_score`: Similarity threshold (range: 0.1-0.99, default: 0.5)
- `dimensions`: Embedding dimensions (1536 for OpenAI models)

## 🎯 Testing Your Implementation

### Run the app

From the repository root:

```bash
# Option A (recommended)
python -m task.app

# Option B
python run_rag.py
```

If your database is not on the defaults, you can override it with environment variables:

```bash
export RAG_PG_HOST=localhost
export RAG_PG_PORT=55432
export RAG_PG_DATABASE=vectordb
export RAG_PG_USER=postgres
export RAG_PG_PASSWORD=postgres
```

### Valid request samples

Prompt 1:
>What safety precautions should be taken to avoid exposure to excessive microwave energy?

Prompt 2:
>What is the maximum cooking time that can be set on the DW 395 HCG microwave oven?

Prompt 3:
>How should you clean the glass tray of the microwave oven?

Prompt 4:
>What materials are safe to use in this microwave during both microwave and grill cooking modes?

Prompt 5:
>What are the steps to set the clock time on the DW 395 HCG microwave oven?

Prompt 6:
>What is the ECO function on this microwave and how do you activate it?

Prompt 7:
>What are the specifications for proper installation, including the required free space around the oven?

Prompt 8:
>How does the multi-stage cooking feature work, and what types of cooking programs cannot be included in it?

Prompt 9:
>What should you do if food in plastic or paper containers starts smoking during heating?

Prompt 10:
>What is the recommended procedure for removing odors from the microwave oven?

### Invalid request samples

Prompt 11:
>What do you know about the DIALX community?

Prompt 12:
>What do you think about the dinosaur era? Why did they die?

## 🛠️ Troubleshooting

**Common Issues:**

- **Database connection error:** Ensure Docker container is running (`docker-compose up -d`)
- **Empty API key:** Update `_constants.py` with your DIAL API key
- **No connection to LLM:** Ensure EPAM VPN is active
- **Vector dimension mismatch:** Ensure embedding dimensions match database schema (1536)

----

<img src="dialx-banner.png">
