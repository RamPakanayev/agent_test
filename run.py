# Run script for Tesla Knowledge Graph RAG Demonstration

import sys
import os
import traceback
import time

# Ensure the 'src' directory is on Python's path so submodules import correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.graph.knowledge_graph import KnowledgeGraph
from src.rag.rag_system import RAGSystem, Document
from src.agents.react_agent import ReActAgent
from src.example import create_sample_documents

if __name__ == "__main__":
    print("🚀 Running ReActAgent on existing Neo4j graph and FAISS index")
    try:
        # Load existing knowledge graph (Neo4j or NetworkX fallback)
        kg = KnowledgeGraph()

        # Initialize RAG system with loaded graph and existing FAISS index
        rag = RAGSystem(kg)
        
        # Load the documents into the RAG system
        print("📄 Loading documents into RAG system...")
        documents = create_sample_documents()
        for doc in documents:
            # Just add docs to memory without reindexing
            if doc.embedding is None:
                doc.embedding = rag._get_embedding_with_retry(doc.content)
            rag.documents.append(doc)
        print(f"✅ Loaded {len(rag.documents)} documents")

        # Create the ReAct agent
        agent = ReActAgent(rag)

        # Example queries to test different capabilities of the system:
        # - Single hop: Direct relationships retrievable from a single graph edge
        # - Multi hop: Requires traversing multiple relationships in the graph
        # - "I don't know": Tests system's ability to recognize when it lacks information
        example_queries = [
            # single hop question:
            "Who is the CEO of Tesla?",
            "Where is Tesla headquartered?",
            "Who founded OpenAI?",
            # multi hop question:
            "who founded the company that headquarters is in the capital of Texas?",
            # i dont know question:
            "what is the capital of France?",
            # most complex question multi-hop:
            "What year was SpaceX founded?"
        ]

        print("\n❓ Processing queries...")
        for query in example_queries:
            print(f"\n{'='*60}\nQUERY: {query}\n{'='*60}")
            response = agent.process_query(query)
            print(f"\n🎯 FINAL RESPONSE: {response}")
            time.sleep(1)

        # Clean up resources
        print("\n🔄 Cleaning up resources...")
        kg.close()
        print("\n✅ Agent run completed successfully!")
    except Exception as e:
        print(f"Error running the agent: {e}")
        traceback.print_exc(file=sys.stdout) 