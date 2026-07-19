"""
All interaction with ChromaDB lives here, and only here. Nothing else in the
codebase should import chromadb directly -- if the vector store is ever
swapped out (e.g. for Pinecone or Weaviate), this is the only file that
needs to change.
"""

# Some Linux/Windows environments ship an outdated system sqlite3 that
# ChromaDB's newer versions refuse to run on. If you hit a sqlite version
# error, run `pip install pysqlite3-binary` and this shim will swap it in
# automatically -- otherwise it's a no-op.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import json
import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_RESULTS_PER_QUERY

COLLECTION_NAME = "sports_history"
CHROMA_PATH = "./chroma_db"


def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client saving to disk."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection(client):
    # ChromaDB's default embedding function runs sentence-transformers
    # locally -- no extra API key or network call needed for embeddings.
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def setup_and_populate_db(json_file_path="./data/sports_facts.json"):
    """
    Reads the offline JSON facts, creates a collection, and populates it.
    Safe to call on every app startup -- it's a no-op once the collection
    already has data, so re-running the app doesn't re-embed everything.
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    if collection.count() > 0:
        print(f"Database already populated with {collection.count()} facts.")
        return collection

    if not os.path.exists(json_file_path):
        print(f"Error: Raw fact data file not found at {json_file_path}")
        return collection

    with open(json_file_path, "r") as f:
        facts_list = json.load(f)

    documents, metadata_list, ids = [], [], []
    for idx, item in enumerate(facts_list):
        documents.append(item["fact"])
        # Storing sport as metadata lets us filter queries by sport later.
        metadata_list.append({"sport": item["sport"]})
        ids.append(f"fact_{idx}")

    collection.add(documents=documents, metadatas=metadata_list, ids=ids)
    print(f"Successfully vectorized and stored {len(documents)} facts.")
    return collection


def query_historic_facts(sport, query_text, n_results=CHROMA_RESULTS_PER_QUERY):
    """
    Queries ChromaDB for historic documents relating to a sport, filtered
    to only that sport's facts via metadata.
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"sport": sport},
    )
    return results.get("documents", [[]])[0]


def add_fresh_fact(sport, fact_text):
    """
    Writes a newly retrieved (web-sourced) fact back into ChromaDB so future
    requests for the same sport can retrieve it from the local store instead
    of re-searching the web every time. This is what keeps the knowledge
    base growing instead of just being a static seed file.
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    # Use a stable id so re-adding the same fact overwrites rather than
    # duplicates it (Chroma's add() upserts on matching ids).
    fact_id = f"web_{sport.lower()}_{abs(hash(fact_text)) % 10**8}"
    collection.add(
        documents=[fact_text],
        metadatas=[{"sport": sport, "source": "web"}],
        ids=[fact_id],
    )
