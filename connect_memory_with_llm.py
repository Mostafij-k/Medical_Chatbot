import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup LLM 
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.5
    )

# ---------------------------
# Prompt Template
# ---------------------------
custom_prompt_template = """
You are a helpful medical assistant.

Use ONLY the context below to answer the question.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template
    )

# Load FAISS DB
DB_FAISS_PATH = "vectoreDB/FAISS_DB"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt()
    }
)

# Run Query
if __name__ == "__main__":
    user_query = input("Write a query here: ")

    response = qa_chain.invoke({"query": user_query})

    print("\nRESULT:\n", response["result"])
    print("\nSOURCE DOCUMENTS:\n", response["source_documents"])