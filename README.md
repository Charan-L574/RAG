# 🌍 OmniDoc AI: Multilingual Intelligent Document Conversational Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)
[![LangChain](https://img.shields.io/badge/🦜-LangChain-green)](https://langchain.com/)

A powerful multilingual conversational AI that can read, understand, and interact with any kind of document, including PDFs with scanned images, text documents, research papers, resumes, and reports. Built with LangChain and Hugging Face APIs, featuring OCR support, multilingual embeddings, contextual document classification, and dynamic response generation.

---

## 🌟 Key Features

### Core Capabilities

- **🌐 Universal Document Support**: Process PDF, DOCX, TXT, CSV, XLSX, PPTX, JPG, PNG files
- **👁️ Intelligent OCR**: Automatic text extraction from scanned documents and images using TrOCR
- **🌍 Multilingual Support**: 
  - Support for 100+ languages
  - Focus on Indian languages: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu
  - Cross-language queries (ask in Spanish, get answer from English document)
  - Automatic language detection and translation
- **🧠 Context-Aware Intelligence**: Adapts behavior based on document type
  - Resume/CV analysis
  - Research paper summarization
  - Legal document simplification
  - Invoice/financial report extraction
  - Textbook content explanation
- **🔍 Advanced RAG**: Retrieval-Augmented Generation with multilingual embeddings
- **💡 Auto-Generated Insights**: Summaries, key points, and suggested questions
- **📊 Document Classification**: Zero-shot classification into 6+ categories
- **🔒 Privacy-Aware**: PII detection and masking
- **💬 Conversational Memory**: Follow-up questions with context awareness
- **📚 Source Citations**: Transparent answers with document references

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Hugging Face API key (free tier available)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your Hugging Face API key
# Get your key from: https://huggingface.co/settings/tokens
```

5. **Run the application**
```bash
python app.py
```

6. **Open your browser**
Navigate to `http://localhost:7860`

---

## ⚙️ Configuration

### Environment Variables

Edit the `.env` file to configure the application:

```env
# Required: Your Hugging Face API Key
HUGGINGFACE_API_KEY=your_api_key_here

# Model Configuration (Optional - defaults provided)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=tiiuae/falcon-7b-instruct
OCR_MODEL=microsoft/trocr-base-printed
CLASSIFICATION_MODEL=facebook/bart-large-mnli
LANGUAGE_DETECTION_MODEL=papluca/xlm-roberta-base-language-detection
TRANSLATION_MODEL=Helsinki-NLP/opus-mt-mul-en

# Application Settings (Optional)
MAX_UPLOAD_SIZE_MB=50
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=3
```

### Getting a Hugging Face API Key

1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up for a free account
3. Navigate to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Create a new token with "Read" permission
5. Copy the token to your `.env` file

---

## 📖 Usage Guide

### Basic Workflow

1. **Upload Documents**
   - Click "Upload Files" and select one or multiple documents
   - Supported formats: PDF, DOCX, TXT, CSV, XLSX, PPTX, JPG, PNG
   - Click "Process Documents"

2. **Review Auto-Generated Insights**
   - Document type classification
   - Language detection
   - Summary and key points
   - Suggested questions

3. **Ask Questions**
   - Type your question in any language
   - Enable/disable translation as needed
   - View answers with source citations

4. **Follow-Up Questions**
   - Ask follow-up questions referencing previous answers
   - Conversation memory maintains context

### Example Use Cases

#### 📄 Resume Analysis
```
Upload: resume.pdf
Ask: "What are the candidate's top 5 skills?"
Ask: "How many years of Python experience does this person have?"
Ask: "Summarize the candidate's work history"
```

#### 📚 Research Paper Understanding
```
Upload: research_paper.pdf
Ask: "What is the main research question?"
Ask: "Explain the methodology used"
Ask: "What are the key findings?"
```

#### 📖 Scanned Textbook
```
Upload: scanned_chapter.jpg
Ask: "What are the main concepts covered?"
Ask: "Explain the first example"
Ask: "List all the definitions"
```

#### 🌐 Multilingual Queries
```
Upload: english_document.pdf
Ask in Hindi: "इस दस्तावेज़ का सारांश क्या है?"
Get answer in Hindi based on English document
```

#### 📊 Invoice Processing
```
Upload: invoice.pdf
Ask: "What is the total amount?"
Ask: "Who is the vendor?"
Ask: "When is the payment due?"
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio UI (app.py)                   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐    ┌──────────▼──────────┐
│   Document     │    │   Multilingual      │
│   Processor    │    │   Processor         │
│  (pipeline.py) │    │ (multilingual.py)   │
└───────┬────────┘    └──────────┬──────────┘
        │                        │
        │             ┌──────────▼──────────┐
        │             │   RAG Engine        │
        └────────────►│  (rag_engine.py)    │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Advanced Features  │
                      │ (advanced_features) │
                      └─────────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │  Hugging Face APIs  │
                      └─────────────────────┘
```

### Data Flow

1. **Document Upload** → `DocumentProcessor` extracts text (with OCR if needed)
2. **Classification** → Zero-shot classifier determines document type
3. **Language Detection** → Identifies document language
4. **Chunking & Embedding** → Text split into chunks, multilingual embeddings generated
5. **Vector Storage** → FAISS index stores embeddings
6. **Query Processing**:
   - Detect query language
   - Translate if needed
   - Retrieve relevant chunks
   - Generate context-aware prompt
   - Call LLM for answer
   - Translate answer back if needed

---

## 🔧 Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual semantic search |
| **LLM** | `tiiuae/falcon-7b-instruct` | Answer generation |
| **OCR** | `microsoft/trocr-base-printed` | Text extraction from images |
| **Classification** | `facebook/bart-large-mnli` | Document type classification |
| **Language Detection** | `papluca/xlm-roberta-base-language-detection` | Auto-detect input language |
| **Translation** | `Helsinki-NLP/opus-mt-*` | Cross-language translation |

### Key Technologies

- **LangChain**: RAG orchestration and text processing
- **Hugging Face**: API access to state-of-the-art models
- **FAISS**: Fast similarity search for vector retrieval
- **Gradio**: Interactive web interface
- **PyPDF2 & pdfplumber**: PDF text extraction
- **python-docx**: DOCX processing
- **Pillow**: Image processing

---

## 📁 Project Structure

```
rag/
├── app.py                    # Main Gradio application
├── pipeline.py               # Document processing & OCR
├── rag_engine.py            # RAG system with embeddings
├── multilingual.py          # Language detection & translation
├── advanced_features.py     # Question gen, PII masking, insights
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🎯 Advanced Features

### 1. Automatic Question Generation
After uploading a document, the system suggests 5 relevant questions you can ask.

### 2. PII Detection and Masking
Automatically detects and masks:
- Email addresses
- Phone numbers
- Credit card numbers
- Dates
- Social security numbers

### 3. Context-Aware Prompting
Different prompt templates for:
- Resumes (focus on skills, experience)
- Research papers (focus on methodology, findings)
- Legal documents (simplify legal language)
- Invoices (focus on numbers, dates)
- Textbooks (educational explanations)

### 4. Document Comparison
Upload multiple documents and compare:
- Document types
- Size comparison
- Content themes

### 5. Conversation Memory
- Maintains last 10 interactions
- Allows follow-up questions
- Context-aware responses

### 6. Source Citations
Every answer includes:
- Source document name
- Page number
- Relevance score
- Content excerpt

---

## 🌐 Supported Languages

### Primary Support (with dedicated translation models)
- English
- Hindi (हिन्दी)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Bengali (বাংলা)
- Marathi (मराठी)
- Gujarati (ગુજરાતી)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- Punjabi (ਪੰਜਾਬੀ)
- Urdu (اردو)

### Additional Support
Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, and 80+ more languages through multilingual embeddings.

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: "HUGGINGFACE_API_KEY not found"
- **Solution**: Make sure you've created a `.env` file and added your API key

**Issue**: OCR not working on scanned PDFs
- **Solution**: Ensure the image quality is good. Try using `.jpg` or `.png` format directly

**Issue**: Slow response times
- **Solution**: Hugging Face API can be slow on free tier. Consider:
  - Reducing `TOP_K_RETRIEVAL` in `.env`
  - Using smaller documents
  - Upgrading to Hugging Face Pro

**Issue**: Translation not working
- **Solution**: Some language pairs may not have pre-trained models. The system falls back to English.

**Issue**: Out of memory errors
- **Solution**: 
  - Reduce `CHUNK_SIZE` in `.env`
  - Process fewer documents at once
  - Use smaller files

---

## 🚀 Performance Tips

1. **Optimize Chunk Size**: Adjust `CHUNK_SIZE` based on your documents
   - Technical docs: 300-500 words
   - Narratives: 500-800 words

2. **Batch Processing**: Process multiple small documents together for efficiency

3. **Cache Results**: The system maintains embeddings in memory during the session

4. **Use Specific Questions**: More specific questions get better answers

5. **Enable Translation Selectively**: Disable translation for English-only workflows

---

## 🔮 Future Enhancements (V2)

- [ ] Real-time collaboration (multi-user chat)
- [ ] Voice input via Speech-to-Text
- [ ] Export Q&A logs to PDF
- [ ] Document version comparison
- [ ] Knowledge graph visualization with NetworkX
- [ ] Local model support (no API required)
- [ ] Custom model fine-tuning
- [ ] Batch document processing API
- [ ] Advanced analytics dashboard
- [ ] Plugin system for custom document types

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing amazing models and APIs
- [LangChain](https://langchain.com/) for the RAG framework
- [Gradio](https://gradio.app/) for the UI framework
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research for vector search

---

## � Project Structure

The project is now organized into clean folders:

```
rag/
├── 📄 Core Application (Python files in root)
│   ├── app.py                    # Main Gradio application
│   ├── rag_engine.py            # RAG engine with embeddings & LLM
│   ├── pipeline.py              # Document processing
│   ├── multilingual.py          # Multilingual support
│   └── advanced_features.py     # OCR, PII detection, etc.
│
├── 📚 Documentation (docs/)
│   ├── QUICKSTART.md            # Quick 5-minute setup
│   ├── INSTALLATION.md          # Detailed installation
│   ├── COMPETITIVE_ADVANTAGES.md # Why RAG vs GPT-4/Claude/Perplexity
│   ├── FAQ_2025.md              # Comprehensive Q&A
│   ├── TECHNICAL_SPECIFICATIONS.md
│   └── [More documentation...]
│
└── 🧪 Tests (tests/)
    ├── test_llm.py              # LLM tests
    └── test_setup.py            # Setup verification
```

**See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete file organization.**

### 📖 Documentation Quick Links

- **New users?** → [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Need to pitch/sell?** → [docs/COMPETITIVE_ADVANTAGES.md](docs/COMPETITIVE_ADVANTAGES.md) ⭐
- **Technical details?** → [docs/TECHNICAL_SPECIFICATIONS.md](docs/TECHNICAL_SPECIFICATIONS.md)
- **All documentation** → [docs/README.md](docs/README.md)

---

## �📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section in [docs/FAQ_2025.md](docs/FAQ_2025.md)
- Review Hugging Face API documentation

---

## ⚠️ Disclaimer

This application uses Hugging Face's API and requires an active internet connection. Response times and quality depend on:
- API availability and rate limits
- Model performance
- Document complexity
- Network speed

For production use, consider:
- Upgrading to Hugging Face Pro
- Implementing local models
- Adding caching layers
- Rate limiting and error handling

---

## 📊 Example Screenshots

### Document Upload & Processing
Upload any document and get instant classification, language detection, and insights.

### Multilingual Chat Interface
Ask questions in your native language and get accurate answers with source citations.

### Auto-Generated Questions
Smart question suggestions based on document content and type.

### Document Insights
Automatic summaries, key points, and statistics for quick understanding.

---

**Built with ❤️ using LangChain + Hugging Face**

*Making documents accessible and conversational in every language!*
