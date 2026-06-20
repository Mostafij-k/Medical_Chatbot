import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Mention Vector DB path
DB_FAISS_PATH = "vectoreDB/FAISS_DB"

# Load Vector Store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db

# Set Custom Prompt
def set_custom_prompt():
    template = """
You are a helpful medical assistant.

Use ONLY the context below to answer the question.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Load LLM Model (GROQ)
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.5
    )

# Create Main App
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="🧠")
    st.title("🧠 Medical Chatbot")

    # session memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # input box
    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        # show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # load vector DB
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("Vector store not loaded!")
                return

            # QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt()
                }
            )

            # get response
            response = qa_chain.invoke({"query": prompt})

            result = response["result"]

            # show assistant response
            st.chat_message("assistant").markdown(result)

            st.session_state.messages.append(
                {"role": "assistant", "content": result}
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")


# run app
if __name__ == "__main__":
    main()