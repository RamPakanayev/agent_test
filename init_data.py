import sys
import os
from pathlib import Path

# Ensure the src directory is in the Python path
dirs = [os.path.join(os.path.dirname(__file__), "src"), os.getcwd()]
for d in dirs:
    if d not in sys.path:
        sys.path.insert(0, d)

from src.example import create_sample_knowledge_graph, create_sample_documents
from src.rag.rag_system import RAGSystem


def main():
    """Initialize the Neo4j graph and FAISS index with sample data"""
    print("🔄 Initializing sample data in Neo4j and FAISS index...")
    kg = create_sample_knowledge_graph()
    rag = RAGSystem(kg)

    documents = create_sample_documents()
    for doc in documents:
        print(f"  - Adding document: {doc.metadata['source']}")
        rag.add_document(doc)

    kg.close()
    print("✅ Initialization complete.")


if __name__ == "__main__":
    main() 