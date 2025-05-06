import sys
import os
from pathlib import Path

# Ensure the src directory is in the Python path
dirs = [os.path.join(os.path.dirname(__file__), "src"), os.getcwd()]
for d in dirs:
    if d not in sys.path:
        sys.path.insert(0, d)

from src.graph.knowledge_graph import KnowledgeGraph, Node, Edge
from src.rag.rag_system import RAGSystem, Document
from src.utils.embeddings import get_embedding


def create_sample_knowledge_graph() -> KnowledgeGraph:
    """Create a sample Tesla-related knowledge graph."""
    kg = KnowledgeGraph()

    print("Creating 'Tesla' node...")
    kg.add_node(Node(
        id="tesla",
        type="company",
        properties={
            "name": "Tesla",
            "description": "An American electric vehicle and clean energy company."
        },
        embedding=get_embedding("Tesla electric car company")
    ))

    print("Creating 'Elon Musk' node...")
    kg.add_node(Node(
        id="elon_musk",
        type="person",
        properties={
            "name": "Elon Musk",
            "description": "CEO of Tesla and founder of SpaceX."
        },
        embedding=get_embedding("Elon Musk CEO Tesla")
    ))

    print("Creating 'Austin' node...")
    kg.add_node(Node(
        id="austin",
        type="location",
        properties={
            "name": "Austin",
            "description": "Headquarters of Tesla."
        },
        embedding=get_embedding("Austin headquarters Tesla")
    ))

    print("Creating 'SpaceX' node...")
    kg.add_node(Node(
        id="spacex",
        type="company",
        properties={
            "name": "SpaceX",
            "description": "A private aerospace company founded by Elon Musk."
        },
        embedding=get_embedding("SpaceX rocket company")
    ))

    # Add relationships
    print("Adding relationships between nodes...")
    kg.add_edge(Edge(
        source="tesla",
        target="elon_musk",
        type="hasCEO",
        properties={"description": "Elon Musk is the CEO of Tesla."}
    ))

    kg.add_edge(Edge(
        source="tesla",
        target="austin",
        type="headquarteredIn",
        properties={"description": "Tesla is headquartered in Austin."}
    ))

    kg.add_edge(Edge(
        source="elon_musk",
        target="spacex",
        type="founded",
        properties={"description": "Elon Musk founded SpaceX."}
    ))

    return kg


def create_sample_documents():
    """Create Tesla-related documents for the RAG system."""
    return [
        Document(
            content="""Elon Musk is the CEO of Tesla, a company that produces electric vehicles
            and is known for its innovation in battery technology and autopilot software.""",
            metadata={
                "source": "wiki_tesla.txt",
                "entities": ["elon_musk", "tesla"],
                "kg_links": {"elon_musk": "person", "tesla": "company"}
            }
        ),
        Document(
            content="""Tesla's headquarters are located in Austin, Texas. The company has factories in
            multiple countries and continues to expand its operations worldwide.""",
            metadata={
                "source": "tesla_about.txt",
                "entities": ["tesla", "austin"],
                "kg_links": {"tesla": "company", "austin": "location"}
            }
        ),
        Document(
            content="""Elon Musk founded multiple companies including SpaceX, Neuralink, and OpenAI.
            His work with Tesla has transformed the electric vehicle industry.""",
            metadata={
                "source": "elon_musk_bio.txt",
                "entities": ["elon_musk", "tesla", "spacex"],
                "kg_links": {"elon_musk": "person", "spacex": "company", "tesla": "company"}
            }
        )
    ]


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