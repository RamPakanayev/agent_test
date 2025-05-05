from graph.knowledge_graph import KnowledgeGraph, Node, Edge
from rag.rag_system import RAGSystem, Document
from agents.react_agent import ReActAgent
from utils.embeddings import get_embedding
import time
from typing import List, Dict, Optional
import json
import sys
import traceback

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

def create_sample_documents() -> List[Document]:
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
    """Main function to demonstrate Graph-RAG with Tesla data."""
    try:
        print("\n" + "="*60)
        print("🚀 STARTING TESLA KNOWLEDGE GRAPH RAG DEMONSTRATION")
        print("="*60)
        
        print("\n📊 Creating Tesla knowledge graph...")
        kg = create_sample_knowledge_graph()

        print("\n🔄 Initializing RAG system...")
        rag = RAGSystem(kg)

        print("\n🧠 Initializing ReAct agent...")
        # Create agent with default settings (chain of thought display is now built-in)
        agent = ReActAgent(rag)

        print("\n📄 Adding Tesla documents...")
        documents = create_sample_documents()
        for i, doc in enumerate(documents):
            try:
                print(f"  - Adding document {i+1}/{len(documents)}: {doc.metadata['source']}...")
                rag.add_document(doc)
                time.sleep(1)  # avoid rate-limiting on embedding API
            except Exception as e:
                print(f"❌ Error adding document: {e}")

        example_queries = [
            # "Who is the CEO of Tesla?",
            # "Where is Tesla headquartered?",
            "who founded openai?",
            # "What is the capital of France?",
            # "What is the relationship between Tesla and innovation?"
            # "How does Tesla's innovation strategy compare to its competitors? and how does it related to aliens?"
            # "how much is 2+2?"
        ]

        print("\n❓ Processing Tesla queries...")
        for query in example_queries:
            try:
                print(f"\n{'='*60}")
                print(f"QUERY: {query}")
                print(f"{'='*60}")
                
                # Process query - will display reasoning chain automatically
                response = agent.process_query(query)
                
                print(f"\n🎯 FINAL RESPONSE: {response}")
                print(f"{'='*60}")
                
                time.sleep(1)
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print(f"Exception details: {type(e).__name__}: {str(e)}")
                traceback.print_exc(file=sys.stdout)

        # Cleanly close Neo4j driver
        print("\n🔄 Cleaning up resources...")
        kg.close()
        
        print("\n✅ Demonstration completed successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred in main: {e}")
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    main()






# # -----OLD CODE------------------------
# # run_rag_tesla_demo.py (Main File)
# # -----------------------------

# from knowledge_graph_rag.src.graph.knowledge_graph import KnowledgeGraph, Node, Edge
# from knowledge_graph_rag.src.rag.rag_system import RAGSystem, Document
# from knowledge_graph_rag.src.agents.react_agent import ReActAgent
# from knowledge_graph_rag.src.utils.embeddings import get_embedding
# import os
# from openai import OpenAI
# from typing import List, Dict
# import time

# def create_sample_knowledge_graph() -> KnowledgeGraph:
#     """Create a sample Tesla-related knowledge graph."""
#     kg = KnowledgeGraph()

#     # Create nodes
#     kg.add_node(Node(
#         id="tesla",
#         type="company",
#         properties={
#             "name": "Tesla",
#             "description": "An American electric vehicle and clean energy company."
#         },
#         embedding=get_embedding("Tesla electric car company")
#     ))

#     kg.add_node(Node(
#         id="elon_musk",
#         type="person",
#         properties={
#             "name": "Elon Musk",
#             "description": "CEO of Tesla and founder of SpaceX."
#         },
#         embedding=get_embedding("Elon Musk CEO Tesla")
#     ))

#     kg.add_node(Node(
#         id="austin",
#         type="location",
#         properties={
#             "name": "Austin",
#             "description": "Headquarters of Tesla."
#         },
#         embedding=get_embedding("Austin headquarters Tesla")
#     ))

#     # Add relationships
#     kg.add_edge(Edge(
#         source="tesla",
#         target="elon_musk",
#         type="hasCEO",
#         properties={"description": "Elon Musk is the CEO of Tesla."}
#     ))

#     kg.add_edge(Edge(
#         source="tesla",
#         target="austin",
#         type="headquarteredIn",
#         properties={"description": "Tesla is headquartered in Austin."}
#     ))

#     kg.add_edge(Edge(
#         source="elon_musk",
#         target="spacex",
#         type="founded",
#         properties={"description": "Elon Musk founded SpaceX."}
#     ))

#     return kg

# def create_sample_documents() -> List[Document]:
#     """Create Tesla-related documents for the RAG system."""
#     return [
#         Document(
#             content="""Elon Musk is the CEO of Tesla, a company that produces electric vehicles
#             and is known for its innovation in battery technology and autopilot software.""",
#             metadata={"source": "wiki_tesla.txt"}
#         ),
#         Document(
#             content="""Tesla's headquarters are located in Austin, Texas. The company has factories in
#             multiple countries and continues to expand its operations worldwide.""",
#             metadata={"source": "tesla_about.txt"}
#         ),
#         Document(
#             content="""Elon Musk founded multiple companies including SpaceX, Neuralink, and OpenAI.
#             His work with Tesla has transformed the electric vehicle industry.""",
#             metadata={"source": "elon_musk_bio.txt"}
#         )
#     ]

# def main():
#     """Main function to demonstrate Graph-RAG with Tesla data."""
#     try:
#         print("Creating Tesla knowledge graph...")
#         kg = create_sample_knowledge_graph()

#         print("Initializing RAG system...")
#         rag = RAGSystem(kg)

#         print("Initializing ReAct agent...")
#         agent = ReActAgent(rag)

#         print("Adding Tesla documents...")
#         documents = create_sample_documents()
#         for doc in documents:
#             try:
#                 rag.add_document(doc)
#                 time.sleep(1)
#             except Exception as e:
#                 print(f"Error adding document: {e}")

#         example_queries = [
#             "Who is the CEO of Tesla?",
#             # "Where is Tesla headquartered?"
#         ]

#         print("\nProcessing Tesla queries...")
#         for query in example_queries:
#             try:
#                 print(f"\n{'='*50}")
#                 print(f"Query: {query}")
#                 print(f"{'='*50}")
#                 response = agent.process_query(query)
#                 print(f"\nFinal Response: {response}")
#                 time.sleep(1)
#             except Exception as e:
#                 print(f"Error processing query: {e}")
    
#     except Exception as e:
#         print(f"An error occurred in main: {e}")

# if __name__ == "__main__":
#     main()
