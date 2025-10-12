# RAG Document Assistant

An intelligent document processing system that leverages Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from your documents. Built for enterprises and individuals who need to extract insights from large document collections.

## Features

- **Multi-format Support**: Process PDF, DOCX, TXT, CSV, XLSX, PPTX, and image files
- **OCR Capability**: Extract text from scanned documents and images using TrOCR
- **Multilingual**: Support for 100+ languages with cross-language querying
- **Advanced Retrieval**: Query expansion, hybrid search, and intelligent reranking
- **Privacy-Aware**: Automatic PII detection and masking
- **Interactive Interface**: User-friendly Gradio web interface
- **Source Attribution**: All answers include document sources and citations

## Quick Start

### Prerequisites
- Python 3.8 or higher
- HuggingFace API key ([Get one here](https://huggingface.co/settings/tokens))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Charan-L574/RAG.git
cd RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:

Create a `.env` file in the project root:
```
HUGGINGFACE_API_KEY=your_key_here
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:7860`

### Usage

1. Upload one or more documents using the file upload interface
2. Click "Process Documents" to index the content
3. Ask questions about your documents in natural language
4. Receive AI-powered answers with source citations

## Technology Stack

- **Language Model**: Meta-Llama-3-8B-Instruct for answer generation
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 for semantic search
- **Vector Database**: FAISS for efficient similarity search
- **OCR Engine**: Microsoft TrOCR for text extraction
- **Framework**: LangChain for RAG orchestration
- **Interface**: Gradio for web UI
- **API**: HuggingFace Inference API

## Architecture

The system implements a modern RAG pipeline:

1. **Document Processing**: Extracts and chunks text from various formats
2. **Embedding Generation**: Creates vector representations using multilingual models
3. **Vector Storage**: Indexes embeddings in FAISS for fast retrieval
4. **Query Processing**: Expands queries and retrieves relevant chunks
5. **Answer Generation**: Uses LLM with retrieved context to generate answers
6. **Post-Processing**: Adds citations and formats responses

## Project Structure

```
RAG/
├── app.py                    # Main Gradio application
├── rag_engine.py            # Core RAG implementation
├── pipeline.py              # Document processing pipeline
├── multilingual.py          # Multilingual support
├── advanced_features.py     # Advanced capabilities
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── .env                     # Configuration (create this)
├── docs/                    # Documentation
└── README.md               # This file
```

## Key Capabilities

### Document Classification
Automatically identifies document types (resume, research paper, legal document, invoice, etc.) and adapts processing accordingly.

### Context-Aware Prompting
Uses specialized prompts based on document type to provide more relevant and accurate answers.

### Conversation Memory
Maintains context across multiple queries, enabling follow-up questions and natural conversation flow.

### Question Generation
Automatically suggests relevant questions based on document content to help users explore the information.

## Performance Characteristics

- **Processing Speed**: ~2-5 seconds per document (varies by size)
- **Query Response**: ~3-8 seconds (includes retrieval and generation)
- **Supported Languages**: 100+ languages for both documents and queries
- **Maximum Document Size**: 50MB per file (configurable)
- **Concurrent Users**: Supports multiple simultaneous users

## Documentation

For detailed guides and references, see:
- [Quick Start Guide](QUICK_START.md)
- [Project Structure](PROJECT_STRUCTURE.md)
- [Complete Documentation](docs/)


**RAG Document Assistant** - Transforming document intelligence through advanced AI technology.
