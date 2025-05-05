# Knowledge Graph RAG System

A comprehensive AI system that combines Knowledge Graphs, Retrieval-Augmented Generation (RAG), and ReAct agents for intelligent information processing and reasoning.

## Features

- **Knowledge Graph Construction**: Build and maintain a semantic knowledge graph from various data sources
- **RAG Integration**: Enhanced retrieval using vector embeddings and semantic search
- **ReAct Agents**: Intelligent agents that can reason and act based on the knowledge graph
- **Graph Algorithms**: Implementation of various graph algorithms for analysis and traversal
- **Reasoning Capabilities**: Support for different types of reasoning (deductive, inductive, abductive)

## Project Structure

```
knowledge_graph_rag/
├── src/
│   ├── graph/           # Knowledge graph implementation
│   ├── rag/            # RAG system components
│   ├── agents/         # ReAct agents
│   ├── reasoning/      # Reasoning engines
│   └── utils/          # Utility functions
├── tests/              # Test cases
├── data/               # Data storage
└── docs/               # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/knowledge_graph_rag.git
cd knowledge_graph_rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

1. Initialize the knowledge graph:
```python
from src.graph import KnowledgeGraph
graph = KnowledgeGraph()
graph.build_from_data("path/to/data")
```

2. Set up the RAG system:
```python
from src.rag import RAGSystem
rag = RAGSystem(graph)
```

3. Create and use ReAct agents:
```python
from src.agents import ReActAgent
agent = ReActAgent(rag)
response = agent.process_query("What is the relationship between X and Y?")
```

## Development

- Run tests: `pytest tests/`
- Format code: `black src/ tests/`
- Sort imports: `isort src/ tests/`
- Type checking: `mypy src/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 