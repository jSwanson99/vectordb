from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from typing import List, Dict
from pathlib import Path
import chromadb
import os

from log_config import get_logger
from env import get_config

# warnings.filterwarnings(
#    action="ignore",
#    message="unclosed",
#    category=ResourceWarning
# )
# warnings.filterwarnings(
#    action="ignore",
#    category=DeprecationWarning
# )

config = get_config()
CHUNK_OVERLAP = config.chunk_overlap


SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.py', '.yaml', '.yml', '.csv'}
logger = get_logger("Main")


class DocumentProcessor:
    def __init__(self, data_path: str = config.data_dir):
        self.logger = get_logger(f"{__name__}.DocumentProcessor")
        self.data_path = Path(data_path)
        self.documents = []

    def scan_directory(self) -> List[Dict]:
        """Scan directory for supported files"""
        files = []
        for file_path in self.data_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'extension': file_path.suffix,
                    'relative_path': str(file_path.relative_to(self.data_path))
                })
            elif file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.logger.info(f"ignoring {file_path.name}")
        return files

    def read_file(self, file_path: str) -> str:
        """Read file content with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - config.chunk_overlap
        return chunks

    def process_files(self) -> List[Dict]:
        """Process all files in directory"""
        files = self.scan_directory()
        documents = []

        for idx, file_info in enumerate(files):
            self.logger.info(
                f"Processing {idx+1}/{len(files)}: {file_info['name']}")

            content = self.read_file(file_info['path'])
            if not content:
                continue

            # Create chunks
            chunks = self.chunk_text(content)

            # Create document entries for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                doc = {
                    'id': f"{file_info['relative_path']}_{chunk_idx}",
                    'content': chunk,
                    'metadata': {
                        'file_path': file_info['relative_path'],
                        'file_name': file_info['name'],
                        'extension': file_info['extension'],
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    }
                }
                documents.append(doc)

        return documents


def main():
    if not os.path.exists(config.data_dir):
        logger.info(f"Creating data directory at {config.data_dir}")
        os.makedirs(config.data_dir)
        logger.info("Please add documents to the data directory and run again.")
        return

    processor = DocumentProcessor(config.data_dir)

    files = processor.scan_directory()
    if not files:
        logger.info(f"No supported files found in {config.data_dir}")
        logger.info(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    logger.info(f"Found {len(files)} files to process")
    documents = processor.process_files()
    logger.info(f"\nCreated {len(documents)} document chunks")

    if config.remote_db != "":
        logger.info(f"Using remote http client {config.remote_db}")
        chroma_client = chromadb.AsyncHttpClient(host=config.remote_db)
    else:
        logger.info("Using in memory client")
        chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=OllamaEmbeddingFunction(
            url=config.openai_url,
            model_name=config.embedding_model,
        )
    )

    for i in range(0, len(documents), config.batch_size):
        batch = documents[i:i+config.batch_size]

        ids = [doc['id'] for doc in batch]
        texts = [doc['content'] for doc in batch]
        metadatas = [doc['metadata'] for doc in batch]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Added batch {i//config.batch_size +
                                   1}/{(len(documents)-1)//config.batch_size + 1}")

    logger.info(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    logger.info(config)
    main()
