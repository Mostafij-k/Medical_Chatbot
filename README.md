🩺 Medical Chatbot using RAG, LangChain & Llama 3

An AI-powered Medical Chatbot built using Retrieval-Augmented Generation (RAG), LangChain, FAISS, and Llama 3 (via Groq). The chatbot retrieves relevant information from medical PDF documents and generates context-aware responses to user queries.

🚀 Project Workflow
Phase 1: Knowledge Base Creation
Load and process medical PDF documents
Split documents into text chunks
Generate vector embeddings
Store embeddings in a FAISS vector 

Phase 2: Connect Memory with LLM
Configure Llama 3 through Groq API
Connect the LLM with the FAISS vector store
Build a retrieval chain using LangChain

Phase 3: Chatbot Interface
Develop an interactive UI with Streamlit
Load and cache the FAISS vector database
Implement Retrieval-Augmented Generation (RAG) for accurate responses

🛠️ Tools & Technologies
Python
LangChain
Llama 3 (Groq)
FAISS
Hugging Face Embeddings
Streamlit
Retrieval-Augmented Generation (RAG)

✨ Features
Medical document question answering
Semantic search using vector embeddings
Context-aware AI responses
Fast retrieval with FAISS
User-friendly Streamlit interface

This project demonstrates how to build a domain-specific AI assistant by combining Large Language Models with Retrieval-Augmented Generation (RAG) techniques.
