# ---------------------------
# kg.py (Knowledge Graph file with Neo4j and NetworkX fallback)
# ---------------------------
import os
from dotenv import load_dotenv
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel
import numpy as np
from utils.embeddings import get_embedding

# Load environment variables
load_dotenv()

class Node(BaseModel):
    id: str
    type: str
    properties: Dict[str, str]
    embedding: Optional[List[float]] = None

class Edge(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, str]

class KnowledgeGraph:
    def __init__(self, uri=None, user=None, password=None):
        """
        Initialize knowledge graph with Neo4j or fallback to NetworkX.
        
        Args:
            uri: Neo4j URI (optional, defaults to NEO4J_URI env variable)
            user: Neo4j username (optional, defaults to NEO4J_USER env variable)
            password: Neo4j password (optional, defaults to NEO4J_PASSWORD env variable)
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.use_neo4j = False
        self.driver = None
        # Always initialize NetworkX graph and node_embeddings for fallback
        self.graph = nx.DiGraph()
        self.node_embeddings: Dict[str, List[float]] = {}
        
        # First, try to use Neo4j if credentials are available
        if self.uri and self.user and self.password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                print(f"✅ Connected to Neo4j at {self.uri}")
                self.use_neo4j = True
            except Exception as e:
                print(f"⚠️ Neo4j connection failed: {e}. Using NetworkX fallback.")
                self.use_neo4j = False
        else:
            print("⚠️ Neo4j credentials not provided. Using NetworkX fallback.")
        # No need to re-initialize self.graph here; always available

    def close(self):
        """Close any open connections."""
        if self.use_neo4j and self.driver:
            self.driver.close()
            print("✅ Neo4j connection closed")

    def add_node(self, node: Node) -> None:
        """Add a node to the knowledge graph."""
        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    # Create node in Neo4j
                    session.run(
                        """
                        MERGE (n:Entity {id: $id})
                        SET n.type = $type, 
                            n += $properties
                        """,
                        id=node.id, 
                        type=node.type, 
                        properties=node.properties
                    )
                    print(f"✅ Added node '{node.id}' to Neo4j")
                    
                    # Store embedding if available
                    if node.embedding:
                        # Store embeddings as separate property
                        # Note: In production, use a vector index
                        session.run(
                            """
                            MATCH (n:Entity {id: $id})
                            SET n.has_embedding = true
                            """,
                            id=node.id
                        )
                return None
            except Exception as e:
                print(f"⚠️ Error adding node to Neo4j: {e}. Using NetworkX fallback.")
        
        # NetworkX fallback
        self.graph.add_node(
            node.id,
            type=node.type,
            properties=node.properties
        )
        if node.embedding:
            self.node_embeddings[node.id] = node.embedding
        print(f"✅ Added node '{node.id}' to NetworkX")

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the knowledge graph."""
        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (a:Entity {id: $source})
                        MATCH (b:Entity {id: $target})
                        MERGE (a)-[r:RELATION {type: $type}]->(b)
                        SET r += $properties
                        """,
                        source=edge.source, 
                        target=edge.target, 
                        type=edge.type, 
                        properties=edge.properties
                    )
                    print(f"✅ Added {edge.source}-[{edge.type}]->{edge.target} to Neo4j")
                return None
            except Exception as e:
                print(f"⚠️ Error adding edge to Neo4j: {e}. Using NetworkX fallback.")
        
        # NetworkX fallback
        self.graph.add_edge(
            edge.source,
            edge.target,
            type=edge.type,
            properties=edge.properties
        )
        print(f"✅ Added {edge.source}-[{edge.type}]->{edge.target} to NetworkX")

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID."""
        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (n:Entity {id: $id})
                        RETURN n
                        """,
                        id=node_id
                    )
                    record = result.single()
                    if record:
                        node_data = record["n"]
                        properties = dict(node_data.items())
                        # Remove internal properties
                        properties.pop('id', None)
                        properties.pop('type', None)
                        
                        return Node(
                            id=node_id,
                            type=node_data.get('type', 'unknown'),
                            properties=properties,
                            embedding=None  # Embeddings not stored directly
                        )
                    return None
            except Exception as e:
                print(f"⚠️ Error getting node from Neo4j: {e}. Using NetworkX fallback.")
        
        # NetworkX fallback
        if node_id in self.graph:
            node_data = self.graph.nodes[node_id]
            return Node(
                id=node_id,
                type=node_data['type'],
                properties=node_data['properties'],
                embedding=self.node_embeddings.get(node_id)
            )
        return None

    def get_related_entities(self, entity_id: str) -> List[str]:
        """Get all entities directly related to a given entity."""
        if self.use_neo4j:
            try:
                query = """
                MATCH (n:Entity)-[r]-(related)
                WHERE n.id CONTAINS $entity_id
                   OR any(prop IN keys(properties(n))
                          WHERE toString(properties(n)[prop]) CONTAINS $entity_id)
                RETURN DISTINCT related.id AS id
                """
                with self.driver.session() as session:
                    result = session.run(query, entity_id=entity_id)
                    return [record["id"] for record in result]
            except Exception as e:
                print(f"⚠️ Error getting related entities from Neo4j: {e}. Using NetworkX fallback.")
        
        # NetworkX fallback
        related_entities = []
        # Check exact node ID match
        if entity_id in self.graph:
            # Get all neighbor nodes
            for neighbor in self.graph.neighbors(entity_id):
                related_entities.append(neighbor)
            for neighbor in self.graph.predecessors(entity_id):
                related_entities.append(neighbor)
        
        # Also search for partial matches in node IDs and properties
        for node_id, node_data in self.graph.nodes(data=True):
            # Check if entity string appears in the node ID
            if entity_id in node_id.lower():
                related_entities.append(node_id)
                continue
                
            # Check if entity string appears in any property value
            props = node_data.get('properties', {})
            for prop_value in props.values():
                if isinstance(prop_value, str) and entity_id in prop_value.lower():
                    related_entities.append(node_id)
                    break

        return list(set(related_entities))

    def query_direct_relationships(self, rel_type=None) -> List[Dict]:
        """Query relationships directly by type.
        
        Args:
            rel_type: Optional filter for relationship type
        
        Returns:
            List of dictionaries with source, target, and relationship info
        """
        results = []
        
        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    # If rel_type is provided, filter by it
                    if rel_type:
                        query = """
                        MATCH (src:Entity)-[r:RELATION {type: $rel_type}]->(tgt:Entity)
                        RETURN src.id as source_id, src.type as source_type, src.name as source_name,
                               tgt.id as target_id, tgt.type as target_type, tgt.name as target_name,
                               r.type as rel_type
                        """
                        result = session.run(query, rel_type=rel_type)
                    else:
                        # Otherwise get all relationships
                        query = """
                        MATCH (src:Entity)-[r:RELATION]->(tgt:Entity)
                        RETURN src.id as source_id, src.type as source_type, src.name as source_name,
                               tgt.id as target_id, tgt.type as target_type, tgt.name as target_name,
                               r.type as rel_type
                        """
                        result = session.run(query)
                        
                    for record in result:
                        results.append({
                            "source": {
                                "id": record["source_id"],
                                "type": record["source_type"],
                                "name": record["source_name"],
                            },
                            "target": {
                                "id": record["target_id"],
                                "type": record["target_type"],
                                "name": record["target_name"],
                            },
                            "relationship": record["rel_type"]
                        })
                        
                    return results
                    
            except Exception as e:
                print(f"⚠️ Error querying relationships from Neo4j: {e}. Using NetworkX fallback.")
        
        # NetworkX fallback
        for source, target, edge_data in self.graph.edges(data=True):
            # Filter by relationship type if provided
            if rel_type and edge_data.get('type') != rel_type:
                continue
                
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            
            results.append({
                "source": {
                    "id": source,
                    "type": source_data.get('type', 'unknown'),
                    "name": source_data.get('properties', {}).get('name', source),
                },
                "target": {
                    "id": target,
                    "type": target_data.get('type', 'unknown'),
                    "name": target_data.get('properties', {}).get('name', target),
                },
                "relationship": edge_data.get('type', 'unknown')
            })
        
        return results

    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Find all paths between two nodes up to a maximum length."""
        if self.use_neo4j:
            try:
                with self.driver.session() as session:
                    cypher = (
                        """
                        MATCH p = shortestPath((a:Entity {id: $source})-[*1..$max_length]-(b:Entity {id: $target}))
                        RETURN [node in nodes(p) | node.id] as path
                        """
                    )
                    result = session.run(
                        cypher,
                        source=source, target=target, max_length=max_length
                    )
                    return [record["path"] for record in result]
            except Exception as e:
                print(f"⚠️ Error finding paths in Neo4j: {e}. Using NetworkX fallback.")
        
        # NetworkX fallback
        try:
            return list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def semantic_search(self, query: str, top_k: int = 5) -> List[Node]:
        """Perform semantic search using node embeddings."""
        # Get the query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            print("⚠️ Could not get embedding for query")
            return []
            
        if self.use_neo4j:
            try:
                # In a real implementation, you would use a Neo4j vector index or plugin
                # For simplicity, we're returning a static result
                print("⚠️ Vector search not implemented in Neo4j connector")
                return []
            except Exception as e:
                print(f"⚠️ Error in Neo4j semantic search: {e}")
                # Fall back to NetworkX
        
        # NetworkX fallback with vector search
        similarities = []
        for node_id, embedding in self.node_embeddings.items():
            if embedding:
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((node_id, similarity))
                
        # Sort by similarity and return top k nodes
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.get_node(node_id) for node_id, _ in similarities[:top_k]]
