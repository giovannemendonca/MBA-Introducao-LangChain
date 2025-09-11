from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Carregando documentos de uma página web
loader = WebBaseLoader("https://www.unimedceara.com.br/")

# Carregando os documentos
docs = loader.load()

# Dividindo os documentos em pedaços menores
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


# Dividindo os documentos em pedaços menores
chunks = splitter.split_documents(docs)


# Exibindo os pedaços
for chunk in chunks:
    print(chunk.page_content)
    print("\n---\n")