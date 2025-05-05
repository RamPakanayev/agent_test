import pytest
from src.graph.knowledge_graph import KnowledgeGraph, Node, Edge

def test_add_node():
    graph = KnowledgeGraph()
    node = Node(
        id="test_node",
        type="test",
        properties={"name": "Test Node"}
    )
    graph.add_node(node)
    assert "test_node" in graph.graph
    assert graph.graph.nodes["test_node"]["type"] == "test"
    assert graph.graph.nodes["test_node"]["properties"] == {"name": "Test Node"}

def test_add_edge():
    graph = KnowledgeGraph()
    # Add nodes first
    node1 = Node(id="n1", type="test", properties={"name": "Node 1"})
    node2 = Node(id="n2", type="test", properties={"name": "Node 2"})
    graph.add_node(node1)
    graph.add_node(node2)
    
    # Add edge
    edge = Edge(
        source="n1",
        target="n2",
        type="test_relation",
        properties={"weight": "1.0"}
    )
    graph.add_edge(edge)
    
    assert graph.graph.has_edge("n1", "n2")
    assert graph.graph.edges["n1", "n2"]["type"] == "test_relation"
    assert graph.graph.edges["n1", "n2"]["properties"] == {"weight": "1.0"}

def test_get_node():
    graph = KnowledgeGraph()
    node = Node(
        id="test_node",
        type="test",
        properties={"name": "Test Node"}
    )
    graph.add_node(node)
    
    retrieved_node = graph.get_node("test_node")
    assert retrieved_node is not None
    assert retrieved_node.id == "test_node"
    assert retrieved_node.type == "test"
    assert retrieved_node.properties == {"name": "Test Node"}

def test_get_neighbors():
    graph = KnowledgeGraph()
    # Add nodes
    node1 = Node(id="n1", type="test", properties={"name": "Node 1"})
    node2 = Node(id="n2", type="test", properties={"name": "Node 2"})
    node3 = Node(id="n3", type="test", properties={"name": "Node 3"})
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    # Add edges
    edge1 = Edge(source="n1", target="n2", type="rel1", properties={})
    edge2 = Edge(source="n1", target="n3", type="rel2", properties={})
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    
    neighbors = graph.get_neighbors("n1")
    assert len(neighbors) == 2
    assert ("n2", "rel1") in neighbors
    assert ("n3", "rel2") in neighbors

def test_find_paths():
    graph = KnowledgeGraph()
    # Add nodes
    nodes = [
        Node(id=f"n{i}", type="test", properties={"name": f"Node {i}"})
        for i in range(4)
    ]
    for node in nodes:
        graph.add_node(node)
    
    # Add edges
    edges = [
        Edge(source="n0", target="n1", type="rel", properties={}),
        Edge(source="n1", target="n2", type="rel", properties={}),
        Edge(source="n2", target="n3", type="rel", properties={})
    ]
    for edge in edges:
        graph.add_edge(edge)
    
    paths = graph.find_paths("n0", "n3")
    assert len(paths) == 1
    assert paths[0] == ["n0", "n1", "n2", "n3"] 