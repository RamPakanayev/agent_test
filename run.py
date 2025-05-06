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

if __name__ == "__main__":
    print("🚀 Running ReActAgent on existing Neo4j graph and FAISS index")
    try:
        # Load existing knowledge graph (Neo4j or NetworkX fallback)
        kg = KnowledgeGraph()

        # Initialize RAG system with loaded graph and existing FAISS index
        rag = RAGSystem(kg)
        
        # Load documents from vectors file 
        rag.load_documents_from_vectors_file()
        
        # If index loaded successfully but no documents in memory, try to match them
        if rag.index is not None and rag.index.ntotal > 0 and not rag.documents:
            print("⚠️ Documents not in memory but index exists, reconstructing document objects...")
            # Create dummy documents with embeddings from the index
            dummy_docs = []
            for i in range(rag.index.ntotal):
                # Create a document with default content but correct embedding position
                dummy_docs.append(Document(
                    content=f"Document from index position {i}",
                    metadata={"source": f"index_position_{i}"},
                    embedding=None  # Will be retrieved from index during search
                ))
            rag.documents = dummy_docs
            print(f"✅ Created {len(dummy_docs)} document references from index")
        
        # Create the ReAct agent
        agent = ReActAgent(rag)

        # Example queries to test different capabilities of the system:
        # - Single hop: Direct relationships retrievable from a single graph edge
        # - Multi hop: Requires traversing multiple relationships in the graph
        # - "I don't know": Tests system's ability to recognize when it lacks information
        example_queries = [
            # single hop question:
            "Who is the CEO of Tesla?",
            # multi hop question:
            "who founded the company that headquarters is in the capital of Texas?",
            # i dont know question:
            "what is the capital of France?",
  
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