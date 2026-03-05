import os
from pathlib import Path

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""

EMBEDDING_MODEL_DEPLOYMENT = "text-embedding-3-small-1"  # keep consistent between indexing and search
CHAT_MODEL_DEPLOYMENT = "gpt-4o"

# NOTE: The DB schema in `init-scripts/init.sql` is VECTOR(1536) by default.
# Keep this aligned with your DB schema to avoid dimension mismatch errors.
EMBEDDING_DIMENSIONS = 1536

DB_CONFIG = {
    "host": os.getenv("RAG_PG_HOST", "localhost"),
    "port": int(os.getenv("RAG_PG_PORT", "5433")),
    "database": os.getenv("RAG_PG_DATABASE", "vectordb"),
    "user": os.getenv("RAG_PG_USER", "postgres"),
    "password": os.getenv("RAG_PG_PASSWORD", "postgres"),
}


def main():
    """
    Entry point for the microwave RAG assistant.

    Experimental notes:
    - The `EMBEDDING_DIMENSIONS` value must always match the dimension configured in
      PostgreSQL (`VECTOR(n)` in `init-scripts/init.sql`). If you index with one
      dimension (e.g. 384) and later try to search with a different dimension
      (e.g. 385), the database will raise a pgvector "dimension mismatch" error
      when running the similarity query.
    - You should use the **same embedding model** for building the context index
      and for query embeddings. Using different models that share the same
      dimensionality (for example, generating context with `text-embedding-3-small`
      but searching with `text-embedding-005`) will still execute successfully,
      but it compares vectors from different embedding spaces, which usually
      harms retrieval quality even though no exception is raised.
    """
    if not API_KEY.strip():
        print("Missing DIAL API key. Please set the `DIAL_API_KEY` environment variable.")
        return

    embeddings_client = DialEmbeddingsClient(
        deployment_name=EMBEDDING_MODEL_DEPLOYMENT,
        api_key=API_KEY,
    )
    completion_client = DialChatCompletionClient(
        deployment_name=CHAT_MODEL_DEPLOYMENT,
        api_key=API_KEY,
    )
    text_processor = TextProcessor(
        embeddings_client=embeddings_client,
        db_config=DB_CONFIG,
    )

    print("🎯 Microwave RAG Assistant")
    print("="*100)

    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip()
    if load_context.lower().strip() in ['y', 'yes']:
        manual_path = Path(__file__).resolve().parent / "embeddings" / "microwave_manual.txt"
        text_processor.process_text_file(
            file_name=str(manual_path),
            chunk_size=300,
            overlap=40,
            dimensions=EMBEDDING_DIMENSIONS,
            truncate_table=True,
        )

        print("="*100)

    conversation = Conversation()
    conversation.add_message(
        Message(Role.SYSTEM, SYSTEM_PROMPT)
    )

    while True:
        user_request = input("\n➡️ ").strip()

        if user_request.lower().strip() in ['quit', 'exit']:
            print("👋 Goodbye")
            break

        # Step 1: Retrieval
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        context = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_request,
            top_k=5,
            score_threshold=0.5,
            dimensions=EMBEDDING_DIMENSIONS,
        )


        # Step 2: Augmentation
        print(f"\n{'=' * 100}\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")
        augmented_prompt = USER_PROMPT.format(
            context="\n\n".join(context),
            query=user_request,
        )
        conversation.add_message(Message(Role.USER, augmented_prompt))

        print(f"Prompt:\n{augmented_prompt}")


        # Step 3: Generation
        print(f"\n{'=' * 100}\n🤖 STEP 3: GENERATION\n{'-' * 100}")
        ai_message = completion_client.get_completion(
            messages=conversation.get_messages(),
            temperature=0.2,
            max_tokens=600,
        )
        conversation.add_message(ai_message)
        print(f"✅ RESPONSE:\n{ai_message.content}")
        print("=" * 100)

# Notes:
# - Make sure Postgres is running (see `docker-compose.yml`).
# - If you change embedding dimensions, you must also update the DB schema in `init-scripts/init.sql`
#   and re-ingest the document chunks.

# APPLICATION ENTRY POINT
if __name__ == "__main__":
    main()