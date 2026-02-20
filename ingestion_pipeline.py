import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import time

load_dotenv()


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please add your documents.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")  # UTF-8 safe!
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")

    print(f"✓ Loaded {len(documents)} documents")
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=100):
    print(f"\nSplitting documents into {chunk_size}-char chunks (overlap {chunk_overlap})...")
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=100):
    print("\nCreating embeddings in batches and storing in ChromaDB...")
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Initialize Chroma DB
    vectorstore = None
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size

    start_time = time.time()
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        batch_start = time.time()

        # Create or add to vectorstore
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            vectorstore.add_documents(batch)

        batch_time = time.time() - batch_start
        count = vectorstore._collection.count()
        print(f"[Batch {batch_num}/{total_batches}] {len(batch)} chunks embedded in {batch_time:.2f}s | Total in DB: {count}")

    total_time = time.time() - start_time
    print(f"\n✓ Finished embedding {total_chunks} chunks in {total_time:.2f}s ({total_time/60:.2f} mins)")
    print(f"Vector database saved to {persist_directory}")
    return vectorstore


def main():
    print("=== RAG Document Ingestion Pipeline with FastEmbed ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. Loading...")
        embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split documents
    chunks = split_documents(documents)

    # Step 3: Embed in batches with progress tracking
    vectorstore = create_vector_store(chunks, persist_directory=persistent_directory)

    print("\n✅ Ingestion complete! Your documents are ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()
