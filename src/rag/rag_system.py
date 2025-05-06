# ---------------------------
# rag.py (RAG System file using FAISS with KG tagging)
# ---------------------------

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import numpy as np
from graph.knowledge_graph import KnowledgeGraph
from utils.embeddings import get_embedding
import faiss
from openai import OpenAI
import time
import os

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RAGSystem:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.documents: List[Document] = []
        self.index = None
        self.index_file = "my_index.faiss"
        self.embedding_retries = 3
        self.retry_delay = 1
        self.dimension = None
        self._load_index()
        self.client = OpenAI()

    def _load_index(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"Loaded FAISS index from {self.index_file}.")
        else:
            self.index = None

    def _get_embedding_with_retry(self, text: str) -> Optional[List[float]]:
        """Try to get embedding with retries on failure."""
        for attempt in range(self.embedding_retries):
            try:
                return get_embedding(text)
            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.embedding_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def add_document(self, document: Document) -> None:
        """Add a document to the RAG system."""
        print(f"Adding document: {document.content[:50]}...")
        if document.embedding is None:
            document.embedding = self._get_embedding_with_retry(document.content)
        self.documents.append(document)
        self._update_index()

    def _update_index(self) -> None:
        """Update the FAISS index with document embeddings."""
        valid_docs = [doc for doc in self.documents if doc.embedding is not None]
        if not valid_docs:
            print("No valid document embeddings available for indexing")
            return
        embeddings = [doc.embedding for doc in valid_docs]
        self.dimension = len(embeddings[0])
        if self.index is None or self.index.ntotal == 0:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # Re-create index if dimension changes
            if self.index.d != self.dimension:
                self.index = faiss.IndexFlatL2(self.dimension)
        self.index.reset()
        self.index.add(np.array(embeddings, dtype=np.float32))
        faiss.write_index(self.index, self.index_file)
        print(f"Saved FAISS index to {self.index_file}")
        self._save_vectors_pretty(valid_docs)

    def _save_vectors_pretty(self, docs: List[Document]):
        """Save vectors and their metadata to a pretty text file."""
        with open("faiss_vectors.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                f.write(f"Document #{i+1}\n")
                f.write(f"Content: {doc.content}\n")
                f.write(f"Metadata: {doc.metadata}\n")
                f.write(f"Embedding: {np.array(doc.embedding)}\n")
                f.write("-"*60 + "\n")
        print("Saved pretty vector data to faiss_vectors.txt")

    def retrieve(self, query: str, top_k: int = 5, relevance_threshold: float = 2.0):
        """Retrieve relevant documents for a query, with a relevance threshold on L2 distance."""
        if not self.documents or self.index is None:
            print("No documents or index available")
            return [], []

        query_embedding = self._get_embedding_with_retry(query)
        if query_embedding is None:
            print("Could not generate query embedding")
            return [], []

        try:
            D, I = self.index.search(
                np.array([query_embedding], dtype=np.float32),
                min(top_k * 2, len(self.documents))  # Get more candidates for filtering
            )
            # Only keep documents with distance below threshold
            docs = []
            dists = []
            for dist, idx in zip(D[0], I[0]):
                if idx < len(self.documents) and dist < relevance_threshold:
                    docs.append(self.documents[idx])
                    dists.append(dist)
            return docs, dists
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            return [], []

    def has_relevant_documents(self, query: str, relevance_threshold: float = 1.0) -> bool:
        """Check if there are any relevant documents for the query."""
        docs, _ = self.retrieve(query, relevance_threshold=relevance_threshold)
        return len(docs) > 0

    def retrieve_with_graph(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve information from both documents and knowledge graph.
        Implements the KG-filtered vector store retrieval approach.
        """
        results = []
        related_kg_entities = []
        
        # Check for direct relationship queries first
        direct_relationship_patterns = {
            "ceo": "hasCEO",
            "who is the ceo": "hasCEO",
            "headquarter": "headquarteredIn",
            "located": "headquarteredIn",
            "found": "founded"
        }
        
        # Try direct relationship query first
        direct_rel_type = None
        for pattern, rel_type in direct_relationship_patterns.items():
            if pattern in query.lower():
                direct_rel_type = rel_type
                break
                
        if direct_rel_type:
            print(f"Detected relationship question about '{direct_rel_type}', querying directly...")
            relationships = self.knowledge_graph.query_direct_relationships(direct_rel_type)
            
            if relationships:
                for rel in relationships:
                    # Create a graph context result
                    results.append({
                        "type": "graph_relationship",
                        "content": f"{rel['source']['name']} {direct_rel_type} {rel['target']['name']}",
                        "metadata": {
                            "source_id": rel['source']['id'],
                            "target_id": rel['target']['id'],
                            "relationship": direct_rel_type
                        }
                    })
                print(f"✅ Found {len(relationships)} direct relationships of type '{direct_rel_type}'")
                # If we found direct relationships, still proceed to add document context
        
        # Step 1: Extract entities from query using naive approach
        # (In production, you'd use NER or entity linking)
        print(f"Extracting entities from query: '{query}'")
        query_entities = []
        for word in query.lower().split():
            word = word.strip("?,.!:;")
            if len(word) > 2:  # Simple heuristic
                query_entities.append(word)
        
        print(f"Extracted query entities: {query_entities}")
        
        # Step 2: Find related entities from KG using the query entities
        try:
            print("Searching knowledge graph for related entities...")
            for entity_id in query_entities:
                # Try to get related entities from KG
                related = self.knowledge_graph.get_related_entities(entity_id)
                if related:
                    print(f"Found related entities for '{entity_id}': {related}")
                    related_kg_entities.extend(related)
                # else:
                    # print(f"No related entities found for '{entity_id}'")
            
            # Always add the query entities themselves
            related_kg_entities.extend(query_entities)
            
            # Remove duplicates
            related_kg_entities = list(set(related_kg_entities))
            print(f"All KG entities to filter with: {related_kg_entities}")
            
            # If we have related entities, try querying node information directly
            if related_kg_entities:
                for entity_id in related_kg_entities:
                    node = self.knowledge_graph.get_node(entity_id)
                    if node:
                        results.append({
                            "type": "graph_node",
                            "content": f"{node.properties.get('name', node.id)}: {node.properties.get('description', '')}",
                            "metadata": {"id": node.id, "type": node.type}
                        })
            
        except Exception as e:
            print(f"Error during KG entity lookup: {e}")
        
        # Step 3: Get vector candidates
        print("Retrieving document candidates using vector search...")
        doc_candidates = self.retrieve(query, top_k * 2)  # Get more candidates for filtering
        print(f"Retrieved {len(doc_candidates[0])} document candidates")
        
        # Step 4: Filter/rerank by KG entities
        print("Filtering document candidates using KG entities...")
        kg_filtered_docs = []
        
        for doc, dist in zip(doc_candidates[0], doc_candidates[1]):
            # Get entities from document metadata
            doc_entities = doc.metadata.get("entities", [])
            
            # Check if there's overlap between document entities and KG entities
            matching_entities = [entity for entity in doc_entities if entity in related_kg_entities]
            
            if matching_entities:
                print(f"Document matches KG entities: {matching_entities}")
                kg_filtered_docs.append((doc, dist))
            else:
                print(f"Document has no matching KG entities")
                
        # Sort by distance (ascending)
        kg_filtered_docs.sort(key=lambda x: x[1])
        
        # Take top_k documents
        filtered_docs = [doc for doc, _ in kg_filtered_docs[:top_k]]
        
        # If we don't have enough documents, supplement with regular vector search results
        if len(filtered_docs) < top_k:
            print(f"Not enough KG-filtered documents, adding {top_k - len(filtered_docs)} unfiltered documents")
            remaining_docs = [doc for doc in doc_candidates[0] if doc not in filtered_docs]
            filtered_docs.extend(remaining_docs[:top_k - len(filtered_docs)])
            
        # Step 5: Create results from documents
        for doc in filtered_docs:
            results.append({
                "type": "document",
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        # If we still have no results, add a fallback result with general info
        if not results:
            general_info = {
                "type": "note", 
                "content": "No specific information found in the knowledge base for this query.",
                "metadata": {}
            }
            results.append(general_info)

        print(f"Final result: {len(results)} items after graph and document retrieval")
        return results

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate a response using the retrieved context."""
        # If context is empty or only contains a note about no context, return "I Don't Know"
        if not context or (
            len(context) == 1 and context[0]["type"] == "note"
        ):
            return "I Don't Know"
        try:
            # Check for direct graph relationship answers
            graph_relationships = [item for item in context if item.get('type') == 'graph_relationship']
            if graph_relationships:
                # For simple relationship questions, we can directly answer
                for rel in graph_relationships:
                    content = rel['content']
                    # Check if content directly answers the query
                    if "hasCEO" in content and ("who" in query.lower() and "ceo" in query.lower()):
                        # Extract the CEO name (format: "Company hasCEO Person")
                        parts = content.split("hasCEO")
                        if len(parts) == 2:
                            return parts[1].strip()
                            
                    elif "headquarteredIn" in content and ("where" in query.lower() and 
                                                         ("headquarter" in query.lower() or "located" in query.lower())):
                        # Extract location (format: "Company headquarteredIn Location")
                        parts = content.split("headquarteredIn")
                        if len(parts) == 2:
                            return parts[1].strip()
                            
                    elif "founded" in content and "found" in query.lower():
                        # Extract founded entity (format: "Person founded Company")
                        parts = content.split("founded")
                        if len(parts) == 2:
                            # If asking who founded
                            if "who" in query.lower():
                                return parts[0].strip()
                            # If asking what was founded
                            else:
                                return parts[1].strip()
            
            # Prepare context string
            context_items = []
            for item in context:
                if item['type'] == 'graph_relationship':
                    # Format relationship data for better readability
                    context_items.append(f"FACT: {item['content']}")
                elif item['type'] == 'graph_node':
                    # Format node data for better readability
                    context_items.append(f"ENTITY: {item['content']}")
                else:
                    context_items.append(f"{item['type'].upper()}:\n{item['content']}")
                    
            context_str = "\n\n".join(context_items)
            
            # Generate response using OpenAI
            print("Generating response with OpenAI...")
            messages = [
                {"role": "system", "content": """You are a knowledgeable AI assistant with expertise on Tesla and Elon Musk.
Use the provided context to answer questions accurately and concisely. 
If the answer is clearly in the context (especially in FACT or ENTITY sections), give a direct, concise answer.
If the answer is not in the context, say: I Don't Know."""},
                {"role": "user", "content": f"""Context:\n{context_str}\n\nQuestion: {query}\n\nPlease provide a clear and accurate response based strictly on the context above. If the answer is not in the context, say: I Don't Know."""}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I Don't Know"

    def process_query(self, query: str) -> str:
        """Process a query and return a response."""
        try:
            # Get context from both documents and knowledge graph
            context = self.retrieve_with_graph(query)
            
            # Generate response
            if context:
                return self.generate_response(query, context)
            else:
                # If no context available, generate a direct response
                return self.generate_response(query, [{
                    "type": "note",
                    "content": "No specific context found. Generating response based on general knowledge.",
                    "metadata": {}
                }])
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"I apologize, but I'm having trouble processing your query. Error: {str(e)}"

    def load_documents_from_vectors_file(self, file_path="faiss_vectors.txt"):
        """Load documents from the vectors file created during index initialization."""
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. No documents loaded.")
            return []

        print(f"Loading documents from {file_path}...")
        current_doc = {}
        docs_loaded = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith("Document #"):
                    # New document section
                    if current_doc.get("content") and current_doc.get("metadata"):
                        # Create a document from previous data
                        try:
                            metadata = eval(current_doc["metadata"])  # Convert string to dict
                            doc = Document(
                                content=current_doc["content"],
                                metadata=metadata,
                                embedding=None  # We'll get from index if needed
                            )
                            self.documents.append(doc)
                            docs_loaded += 1
                        except Exception as e:
                            print(f"Error loading document: {e}")
                    
                    current_doc = {}  # Reset for new document
                
                elif line.startswith("Content:"):
                    current_doc["content"] = line[len("Content:"):].strip()
                
                elif line.startswith("Metadata:"):
                    current_doc["metadata"] = line[len("Metadata:"):].strip()
                
                i += 1
            
            # Process the last document if any
            if current_doc.get("content") and current_doc.get("metadata"):
                try:
                    metadata = eval(current_doc["metadata"])
                    doc = Document(
                        content=current_doc["content"],
                        metadata=metadata,
                        embedding=None
                    )
                    self.documents.append(doc)
                    docs_loaded += 1
                except Exception as e:
                    print(f"Error loading document: {e}")
        
        print(f"✅ Loaded {docs_loaded} documents from vectors file")
        return self.documents
