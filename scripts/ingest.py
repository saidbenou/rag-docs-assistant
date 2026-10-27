"""CLI: python -m scripts.ingest [--dir data/pdfs]"""
import argparse
import logging

from app.ingestion.pipeline import ingest_directory

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Index a folder of PDFs into ChromaDB.")
    parser.add_argument("--dir", default=None, help="PDF folder (default: DATA_DIR)")
    args = parser.parse_args()
    print(ingest_directory(args.dir))
