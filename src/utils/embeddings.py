from typing import List, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import time

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = os.path.join(project_root, '.env')

print(f"Looking for .env file at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# # Load environment variables from .env file
# if os.path.exists(env_path):
#     with open(env_path, 'r', encoding='utf-8-sig') as f:  # Use utf-8-sig to handle BOM
#         print(".env file EXISTS:")
#         # print(f.read())
# else:
#     print("Warning: .env file not found")

# Load environment variables
load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')  # Use utf-8-sig to handle BOM

# Initialize OpenAI client with backoff settings
MAX_RETRIES = 3
RETRY_DELAY = 1

def get_openai_client():
    """Get OpenAI client with API key from environment."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try alternative key names that might have BOM
        api_key = os.getenv("\ufeffOPENAI_API_KEY")
        
    if not api_key:
        print("Environment variables after loading:")
        print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
        print(f"BOM OPENAI_API_KEY: {os.getenv('\ufeffOPENAI_API_KEY')}")
        print("Current working directory:", os.getcwd())
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
        
    return OpenAI(api_key=api_key)

def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get embedding for a text using OpenAI's API with retry logic.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding, or None if an error occurred
    """
    client = get_openai_client()
    
    # Implement retry with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff
                sleep_time = RETRY_DELAY * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All embedding attempts failed")
                return None

def get_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Get embeddings for multiple texts with batching.
    For production use, implement batching to avoid rate limits.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings (or None for failed embeddings)
    """
    # For production use, batch the requests
    embeddings = []
    for i, text in enumerate(texts):
        print(f"Getting embedding {i+1}/{len(texts)}")
        embedding = get_embedding(text)
        embeddings.append(embedding)
        # Add a small delay to avoid rate limiting
        if i < len(texts) - 1:
            time.sleep(0.5)
    return embeddings

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (0-1 where 1 is identical)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_most_similar(query_embedding: List[float], 
                      embeddings: List[List[float]], 
                      top_k: int = 5) -> List[int]:
    """
    Find indices of the most similar embeddings to the query.
    
    Args:
        query_embedding: The query vector
        embeddings: List of embedding vectors to search
        top_k: Number of results to return
        
    Returns:
        List of indices of the most similar embeddings
    """
    similarities = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(embeddings)
        if emb is not None  # Skip None embeddings
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in similarities[:top_k]] 