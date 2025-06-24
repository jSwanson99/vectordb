import openai
import pandas as pd
import os
import json
import requests
from ast import literal_eval
import chromadb
import warnings
import zipfile
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


EMBEDDING_MODEL = "text-embedding-3-small"
DATA_PATH = "./data"
FILE_NAME = "vector_database_wikipedia_articles_embedded.csv"

warnings.filterwarnings(
    action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    if not os.path.exists(f"{DATA_PATH}/{FILE_NAME}"):
        download()
    else:
        print("already downloaded")

    df = pd.read_csv(f"{DATA_PATH}/{FILE_NAME}")
    print(df.info(show_counts=True))

    chroma_client = chromadb.EphemeralClient()
    embedding_function = OllamaEmbeddingFunction(
        url="http://localhost:11434",  # Default Ollama URL
        model_name="nomic-embed-text"  # Or other embedding model
    )

    wikipedia_content_collection = chroma_client.create_collection(
        name='wikipedia_content',
        embedding_function=embedding_function
    )

    wikipedia_title_collection = chroma_client.create_collection(
        name='wikipedia_titles',
        embedding_function=embedding_function
    )

    print(df['content_vector'].iloc[0])
    print(type(df['content_vector'].iloc[0]))

    test_vector = literal_eval(df['content_vector'].iloc[0])
    print(f"Parsed vector length: {
          len(test_vector) if test_vector else 'Failed'}")
    df['content_vector'] = df.content_vector.apply(literal_eval)
    df['title_vector'] = df.title_vector.apply(literal_eval)

    print("adding collections")

    wikipedia_content_collection.add(
        ids=df.vector_id.tolist(),
        embeddings=df.content_vector.tolist(),
    )

    wikipedia_title_collection.add(
        ids=df.vector_id.tolist(),
        embeddings=df.title_vector.tolist(),
    )

    print("done adding collections")

    print(f"Collection created: {wikipedia_title_collection.name}")
    print(f"Collection count: {wikipedia_title_collection.count()}")

    collections = chroma_client.list_collections()
    print([c.name for c in collections])


def download():
    """Loads example content to disk"""

    print("starting download")
    embeddings_url = f"https://cdn.openai.com/API/examples/data/{FILE_NAME}"
    response = requests.get(embeddings_url)
    response.raise_for_status()
    print("finished download successfully")

    with open("file.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("file.zip", "r") as zip_ref:
        zip_ref.extractall(DATA_PATH)


if __name__ == "__main__":
    main()
