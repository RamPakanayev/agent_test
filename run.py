# Run script for Tesla Knowledge Graph RAG Demonstration

import sys
import os
import traceback

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.example import main

if __name__ == "__main__":
    print("Starting Knowledge Graph RAG with Tesla example...")
    try:
        main()
    except Exception as e:
        print(f"Error running the application: {e}")
        traceback.print_exc(file=sys.stdout)
        print("\nCheck that you have set the required environment variables:")
        # print("  - OPENAI_API_KEY: Your OpenAI API key")
        # print("  - NEO4J_URI: Your Neo4j database URI (optional, will use NetworkX fallback)")
        # print("  - NEO4J_USER: Your Neo4j username (optional)")
        # print("  - NEO4J_PASSWORD: Your Neo4j password (optional)") 