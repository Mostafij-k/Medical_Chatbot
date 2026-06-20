from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Load Raw PDF file
Data_path='data/'
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents
documnets=load_pdf_files(data=Data_path)

# Create chunks
def create_chunks(load_data):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    )
    text_chunks=text_splitter.split_documents(load_data)
    return text_chunks
text_chunks=create_chunks(load_data=documnets)
print(len(text_chunks))

# Create Vector Embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L12-v2'
    )
    return embedding_model
embedding_model=get_embedding_model()

#  Store Embedding in FAISS
DB_FAISS_PATH='vectoreDB/FAISS_DB'
db=FAISS.from_documents(
    text_chunks,
    embedding_model
)
db.save_local(DB_FAISS_PATH)