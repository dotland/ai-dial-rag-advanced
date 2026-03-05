from enum import Enum

try:
    from enum import StrEnum  # Python 3.11+
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        """Fallback for Python < 3.11."""

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        try:
            import psycopg2
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing dependency `psycopg2-binary`. Install it with `pip install -r requirements.txt`."
            ) from exc

        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
            self,
            file_name: str,
            chunk_size: int,
            overlap: int,
            dimensions: int,
            truncate_table: bool = True,
    ):
        """
        Load content from file, chunk it, generate embeddings, and save to DB

        Behavior notes for experiments:
        - The embedding vector length **must** match the `VECTOR(n)` dimension in PostgreSQL.
          If, for example, the table stores 384‑dimensional vectors but you try to insert
          a 385‑dimensional embedding, PostgreSQL will raise a runtime error similar to
          "dimension mismatch for type vector" when executing the INSERT.
        - This code does not know which embedding model was used to create existing rows.
          If you re‑ingest the document with a different model (even with the same dimension,
          e.g. `text-embedding-3-small` vs `text-embedding-005` which are both 384‑dimensional),
          the insert will still succeed, but you will now have a mixture of incompatible
          embedding spaces in the same table, which will make similarity scores and retrieval
          quality hard to interpret.
        Args:
            file_name: path to file
            chunk_size: chunk size (min 10 chars)
            overlap: overlap chars between chunks
            dimensions: number of dimensions to store
            truncate_table: truncate table if true
        """

        if chunk_size < 10:
            raise ValueError("chunk_size must be at least 10")
        if overlap < 0:
            raise ValueError("overlap must be at least 0")
        if overlap >= chunk_size:
            raise ValueError("overlap should be lower than chunkSize")

        if truncate_table:
            self._truncate_table()

        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = chunk_text(text=content, chunk_size=chunk_size, overlap=overlap)
        embeddings = self.embeddings_client.get_embeddings(
            inputs=chunks,
            dimensions=dimensions,
            print_request=False,
            print_response=False
        )

        print(f"Processing document: {file_name}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total embeddings: {len(embeddings)}")

        document_name = file_name.split("/")[-1]
        for i in range(len(chunks)):
            embedding = embeddings.get(i)
            if embedding is None:
                raise ValueError(f"Missing embedding for chunk index {i}")
            self._save_chunk(embedding=embedding, chunk=chunks[i], document_name=document_name)


    def _truncate_table(self):
        """Truncate the vectors table"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")
            conn.commit()

    def _save_chunk(self, embedding: list[float], chunk: str, document_name: str):
        """Save chunk with embedding to database"""
        vector_string = f"[{','.join(map(str, embedding))}]"
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, chunk, vector_string)
                )
            conn.commit()



    def search(
            self,
            search_mode: SearchMode,
            user_request: str,
            top_k: int,
            score_threshold: float,
            dimensions: int
    ) -> list[str]:
        """
        Performs similarity search

        Behavior notes for experiments:
        - The query embedding produced for `user_request` must have the same dimension
          as the vectors stored in the `vectors` table. If the stored vectors have 384
          dimensions, but you accidentally request a 385‑dimensional embedding, the
          pgvector similarity operators (`<->` / `<=>`) will raise a runtime error
          when this method executes the SELECT, because the two vectors have different
          lengths.
        - If you keep the **same dimension** but swap to a different embedding model
          for search than the one used to build the index (for example, storing
          context with `text-embedding-3-small` and later querying with
          `text-embedding-005`, both 384‑dimensional), the SQL query will work and
          return results. However, the numerical distances are now between points
          from different embedding spaces, so similarity scores no longer correspond
          cleanly to semantic similarity. In practice this usually means worse or
          unpredictable retrieval quality, even though there is no explicit error.
        Args:
            search_mode: Search mode (Cosine or Euclidian distance)
            user_request: User request
            top_k: Number of results to return
            score_threshold: Minimum score to return (range 0.0 -> 1.0)
            dimensions: Number of dimensions to return (has to be the same as data persisted in VectorDB)
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError("score_threshold must be in [0.0..., 0.99...] range")

        embeddings = self.embeddings_client.get_embeddings(
            inputs=user_request,
            dimensions=dimensions,
            print_request=False,
            print_response=False
        )
        embedding = embeddings.get(0)
        if embedding is None:
            raise ValueError("No embedding returned for the user request")

        vector_string = f"[{','.join(map(str, embedding))}]"

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        retrieved_chunks = []
        with self._get_connection() as conn:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    self._get_search_query(search_mode),
                    (vector_string, vector_string, max_distance, top_k)
                )
                results = cursor.fetchall()

                for row in results:
                    if search_mode == SearchMode.COSINE_DISTANCE:
                        similarity = 1.0 - row['distance']
                    else:
                        similarity = 1.0 / (1.0 + row['distance'])

                    print(f"---Similarity score: {similarity:.2f}---")
                    print(f"Data: {row['text']}\n")
                    retrieved_chunks.append(row['text'])

        return retrieved_chunks

    def _get_search_query(self, search_mode: SearchMode) -> str:
        return """SELECT text, embedding {mode} %s::vector AS distance
                  FROM vectors
                  WHERE embedding {mode} %s::vector <= %s
                  ORDER BY distance
                  LIMIT %s""".format(mode='<->' if search_mode == SearchMode.EUCLIDIAN_DISTANCE else '<=>')
