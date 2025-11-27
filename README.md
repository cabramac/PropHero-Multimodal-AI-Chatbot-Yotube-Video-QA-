🏠 PropHero Multimodal RAG Chatbot
A Retrieval-Augmented Generation (RAG) system that transforms PropHero’s video and blog content into a conversational, intelligent, and factual AI chatbot.
This project integrates speech-to-text, semantic chunking, vector databases, retrievers, and LLM generation using LangChain and Streamlit.

📌 Project Overview

This project builds an end-to-end pipeline to:
- Extract and clean content from YouTube videos and blogs
- Generate structured segment-based transcripts
- Chunk and embed the content using MiniLM-L6-v2
- Store embeddings inside ChromaDB
- Retrieve relevant chunks using multi-query and ranker retrievers
- Generate grounded answers using an LLM
- Evaluate the system using ROUGE and BLEU
- Provide a Streamlit user interface for interaction

The final output is a production-ready RAG chatbot tailored to PropHero’s content.

📁 Repository Structure
📦 PropHero_RAG_Project
│
├── notebook_1_data_integration.ipynb
├── notebook_2_embeddings_and_vectorDB.ipynb
├── notebook_3_RAG_pipeline_and_evaluation.ipynb
├── app.py                       # Streamlit app
├── data/
│   ├── audio/
│   ├── transcripts/
│   ├── chroma_db/               # Vector store
│
├── README.md
└── requirements.txt

🧩 Notebook 1 – Data Integration (Speech Recognition & Cleaning)
✔ Main Objective
Ingest PropHero content (YouTube + blogs) and convert everything into clean, structured text.

✔ Key Features
- Automatic YouTube audio extraction using yt-dlp
- Audio normalization via FFmpeg → MP3
- Speech-to-text transcription using Whisper (local model)
- Segment-level metadata stored:
- start timestamp
- end timestamp
- cleaned text
- video_id
- title
- URL
- Custom cleaning pipeline (removes “[Music]”, “[Applause]”, extra spaces)
- Dual storage:

  .txt → clean global transcript

  .json → structured segments + metadata

✔ Output
Creates a clean dataset ready for chunking and embeddings.

🧩 Notebook 2 – Chunking, Embedding & Vector Database
✔ Main Objective
Transform cleaned text into semantic embeddings and store them in ChromaDB.

✔ Chunking Strategy (LangChain Recursive Splitter)
- YouTube transcripts: chunk size = 600, overlap = 100
- Blogs: chunk size = 800, overlap = 100
- Overlap prevents loss of context across chunk boundaries.

✔ Embedding Model
- sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional vectors
- Lightweight and fast
- Good for sentence-level embeddings

✔ Vector Store
- ChromaDB (persisted locally)
Stores:
  - embeddings
  - chunks
  - metadata
  - Enables cosine similarity search

✔ Output
A fully populated Chroma vector database ready for retrieval.

🧩 Notebook 3 – RAG Pipeline, Retrieval & Evaluation
✔ Main Objective
Build the RAG chain, configure retrievers, and evaluate the system.

🚀 Retrievers Implemented
1. Multi-Query Retriever
- Expands the user query into multiple variations
- Retrieves more complete context
- Reduces the chance of missing relevant chunks

2. Ranker Retriever
- Reorders retrieved chunks by relevance
- Improves precision and reduces noise
- Ensures top-k results are meaningful

🧠 LLM Configuration
Uses the OpenAI API key to generate final answers based on retrieved chunks.
The LLM:
- reads the user question
- reads retrieved context
- produces a grounded, factual answer

📊 Evaluation Metrics Used
ROUGE-L = 0.26 → Good semantic overlap
BLEU = 5.8 → Expectedly low for open-ended RAG (normal)

✔ Output
A fully working RAG QA pipeline with evaluation results.

💻 Streamlit App (UI Layer)
- The app.py file provides a clean interface for user interaction.
Features:
  - Chat interface
  - Typing indicators
  - Conversation history
  - Calls the RAG chain
  - Memory (short window) 
  - Displays grounded, context-aware answers


🧠 Technologies Used
- Whisper (speech-to-text)
- FFmpeg + yt-dlp
- LangChain
- ChromaDB
- SentenceTransformers (MiniLM)
- Streamlit
- OpenAI LLMs
- Python

🏁 Conclusion
This project demonstrates a complete, end-to-end RAG system that integrates multimodal data sources (audio + text), advanced retrieval strategies, and a real LLM to deliver PropHero-specific, factual answers.
It is structured for scalability and can be deployed as a real product with minimal modifications.
