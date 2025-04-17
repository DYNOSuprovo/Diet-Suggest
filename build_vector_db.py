from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# Step 1: Load all PDFs from the 'pdfs' folder
loader = DirectoryLoader("pdfs", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Step 2: Convert them into embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Create and persist the vector store
db = Chroma.from_documents(documents, embedding=embedding, persist_directory="db")
db.persist()

print("✅ Vector store created successfully from all PDFs.")
