# Knowledge Graph RAG System

A comprehensive AI library that combines Knowledge Graphs, Retrieval-Augmented Generation (RAG), and ReAct agents for intelligent information processing and reasoning.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

- Knowledge Graph construction and management
- Retrieval-Augmented Generation with semantic embeddings
- ReAct agents for step-by-step reasoning and decision-making
- Support for in-memory (NetworkX) or Neo4j graph backends
- Utility functions for embeddings and vector storage (FAISS)

## Project Structure

```
.
├── src/
│   ├── graph/          # Knowledge graph implementation (Node, Edge, etc.)
│   ├── rag/            # RAG system components (RAGSystem, Document, etc.)
│   ├── agents/         # ReAct agent implementations
│   └── utils/          # Utility functions (embeddings, dotenv loader, etc.)
├── tests/              # Unit and integration tests
├── run.py              # Demonstration entry point script
├── setup.py            # Package configuration and metadata
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (ignored by Git)
├── faiss_vectors.txt   # Example FAISS vectors file
├── my_index.faiss      # Example FAISS index file
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/knowledge_graph_rag.git
   cd knowledge_graph_rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows (PowerShell):
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

Create a `.env` file in the project root to store your environment variables. This file is ignored by Git.

```ini
OPENAI_API_KEY=<your_openai_api_key>
# Optional Neo4j configuration:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=<your_password>
```

## Usage

### Initialization and Running Separately

The system is designed with separation between data initialization and agent execution:

1. Initialize the knowledge graph and documents:
   ```bash
   python init_data.py
   ```
   This creates the knowledge graph in Neo4j (or NetworkX) and builds the FAISS index.

2. Run the agent on the existing data:
   ```bash
   python run.py
   ```
   This loads the graph and documents (without rebuilding them) and runs queries.

This separation allows for initializing data once and running queries multiple times without rebuilding the knowledge graph or re-embedding documents.

### Running the All-in-One Demonstration

A combined initialization and execution demo is available in `src/example.py`:

```bash
python -m src.example
```

This script will:
1. Build a sample knowledge graph for Tesla.
2. Initialize the RAG system with vector embeddings.
3. Add sample Tesla-related documents.
4. Execute example queries using a ReAct agent.
5. Clean up resources (close graph connection).

### Using as a Library

Import and integrate components in your own Python code:

```python
from src.graph.knowledge_graph import KnowledgeGraph, Node, Edge
from src.rag.rag_system import RAGSystem, Document
from src.agents.react_agent import ReActAgent
from src.utils.embeddings import get_embedding

# Example: build your own graph, add documents, run queries
```

## Running Tests

Execute all tests with pytest:

```bash
pytest tests/
```

## Development

- Format code: `black src/ tests/`
- Sort imports: `isort src/ tests/`
- Type check: `mypy src/`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Make your changes and commit: `git commit -m "Add YourFeature"`
4. Push to your branch: `git push origin feature/YourFeature`
5. Open a Pull Request and describe your changes

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 