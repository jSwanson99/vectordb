import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import sys


DATA_PATH = "./data"


def query_collection(query_text: str, n_results: int = 5):
    """Query the vector database"""
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=f"{DATA_PATH}/chroma_db")

    # Set up embedding function (must match what was used to create embeddings)
    embedding_function = OllamaEmbeddingFunction(
        url="http://ollama:11434",
        model_name="nomic-embed-text"
    )

    # Get collection
    try:
        collection = chroma_client.get_collection(
            name="documents",
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"Error: Could not find collection. Please run main.py first to create embeddings.")
        print(f"Details: {e}")
        return

    # Perform query
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    # Display results
    print(f"\nQuery: '{query_text}'")
    print("=" * 80)

    if not results['documents'][0]:
        print("No results found.")
        return

    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. File: {metadata['file_path']}")
        print(f"   Chunk: {metadata['chunk_index'] +
              1}/{metadata['total_chunks']}")
        print(f"   Distance: {distance:.4f}")
        print(f"   Content:")
        print(f"   {'-' * 75}")
        # Show more content for better context
        print(f"   {doc[:500]}{'...' if len(doc) > 500 else ''}")


def interactive_mode():
    """Run in interactive query mode"""
    print("VectorDB Query Interface")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")

    while True:
        query = input("\nEnter query: ").strip()

        if query.lower() in ['quit', 'exit']:
            break
        elif query.lower() == 'help':
            print("\nCommands:")
            print("  quit/exit - Exit the program")
            print("  help - Show this help")
            print("  stats - Show collection statistics")
            print("  Any other text - Search the vector database")
        elif query.lower() == 'stats':
            show_stats()
        elif query:
            query_collection(query)


def show_stats():
    """Show collection statistics"""
    try:
        chroma_client = chromadb.PersistentClient(
            path=f"{DATA_PATH}/chroma_db")
        collection = chroma_client.get_collection(name="documents")

        print(f"\nCollection Statistics:")
        print(f"Total documents: {collection.count()}")

        # Get sample of metadata to show file distribution
        sample = collection.get(limit=1000)
        if sample['metadatas']:
            files = set(m['file_path'] for m in sample['metadatas'])
            print(f"Files indexed (sample): {len(files)}")
            for f in sorted(files)[:10]:
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
    except Exception as e:
        print(f"Error getting statistics: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line query mode
        query = " ".join(sys.argv[1:])
        query_collection(query)
    else:
        # Interactive mode
        interactive_mode()
