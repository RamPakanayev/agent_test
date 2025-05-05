from setuptools import setup, find_packages
import os

# Read README.md with proper encoding
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="knowledge_graph_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "networkx>=3.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive AI system that combines Knowledge Graphs, RAG, and ReAct agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/knowledge_graph_rag",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 