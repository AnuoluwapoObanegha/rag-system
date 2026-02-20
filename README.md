RAG Project: Document Question Answering

This project is a Retrieval-Augmented Generation (RAG) system. It lets you ask questions about documents you have, and the system finds the relevant parts of your documents and generates answers.

i used:

FastEmbed for creating embeddings (vector representations of text).

ChromaDB to store these embeddings for quick searching.

DeepSeek from OpenRouter as the LLM to generate answers based on retrieved text.

🔹 Features

Load multiple text documents from a folder.

Split large documents into smaller chunks for better searching.

Convert text chunks into embeddings using FastEmbed.

Store embeddings in ChromaDB for fast retrieval.

Search for relevant text using a query.

Generate answers from the retrieved text using DeepSeek (OpenRouter).

🔹 Installation

Run these commands in your terminal (inside a virtual environment or globally):
pip install langchain langchain_community langchain_chroma langchain_text_splitters langchain_fastembed openrouter python-dotenv 

🔹 How It Works

Loading Documents
The system reads all .txt files from a folder called docs.

If files have wrong encoding, they may not load properly.

Splitting Documents
Documents are split into chunks because LLMs can’t handle huge text at once.

Chunking makes it easier to find similarities between text and questions.

Embedding the Chunks
Each chunk is converted into a vector using FastEmbed.

Vectors represent the text in a way the computer can search quickly.

Storing in ChromaDB
All vectors are stored in ChromaDB, which acts like a searchable database of your document knowledge.

Retrieving Relevant Text
When you ask a question, the system searches ChromaDB for the most relevant chunks.

Retrieval uses the same embedding model (FastEmbed) to avoid errors.

Generating Answers
The relevant chunks are sent to DeepSeek (OpenRouter) to generate a natural, human-like answer.

🔹 How to Run

Place all your .txt documents in a folder called docs.

Run the ingestion pipeline to load, chunk, and embed documents:
python ingestion_pipeline.py

Run the retrieval pipeline to ask questions:
python retrieval_pipeline.py

Enter a question like:

How much did Microsoft pay to acquire GitHub?
The system will retrieve relevant chunks and generate an answer.

🔹 To Note

Always use the same embedding model for indexing and retrieval.

Chunk size matters: too big → slow retrieval, too small → too many chunks.

You can explore ChromaDB to see how many chunks are stored.

🔹 Future Improvements

Add support for PDFs or Word documents.

Implement a web interface for easier access.

Improve answer generation with context filtering to remove irrelevant chunks.