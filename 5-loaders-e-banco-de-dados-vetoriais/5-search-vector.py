import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
for k in ("OPENAI_API_KEY", "PGVECTOR_URL","PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")
      
query = "Conte-me mais sobre a avaliação do pensamento gpt-5 e os resultados de desempenho em comparação com o gpt-4"


embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

results = store.similarity_search_with_score(query, k=10)

for i, (doc, score) in enumerate(results):
    print("="*50)
    print(f"Result {i+1} (score: {score:.2f})")
    print("="*50)
    
    print("\nTexto:\n")
    print(doc.page_content.strip())
    
    print("\nMetadados:\n")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")
        