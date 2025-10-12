# Configuration Guide for OmniDoc AI

This guide explains all configuration options and how to optimize them for your use case.

---

## Environment Variables

### Required Configuration

#### HUGGINGFACE_API_KEY
```env
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxx
```
- **Required**: Yes
- **Description**: Your Hugging Face API authentication token
- **Get it from**: https://huggingface.co/settings/tokens
- **Permissions needed**: Read access
- **Free tier**: Yes, with rate limits

---

## Model Configuration

### EMBEDDING_MODEL
```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```
- **Default**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Purpose**: Generate semantic embeddings for document chunks
- **Multilingual**: Yes, supports 50+ languages
- **Dimension**: 384

**Alternatives:**
```env
# Larger, more accurate (slower)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Smaller, faster (less accurate)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### LLM_MODEL
```env
LLM_MODEL=tiiuae/falcon-7b-instruct
```
- **Default**: `tiiuae/falcon-7b-instruct`
- **Purpose**: Generate natural language responses
- **Parameters**: 7B
- **Context length**: ~2048 tokens

**Alternatives:**
```env
# Better for long contexts
LLM_MODEL=google/flan-t5-xxl

# Faster responses
LLM_MODEL=google/flan-t5-base

# More conversational
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.1
```

### OCR_MODEL
```env
OCR_MODEL=microsoft/trocr-base-printed
```
- **Default**: `microsoft/trocr-base-printed`
- **Purpose**: Extract text from scanned images
- **Best for**: Printed text

**Alternatives:**
```env
# For handwritten text
OCR_MODEL=microsoft/trocr-base-handwritten

# For better accuracy (slower)
OCR_MODEL=microsoft/trocr-large-printed
```

### CLASSIFICATION_MODEL
```env
CLASSIFICATION_MODEL=facebook/bart-large-mnli
```
- **Default**: `facebook/bart-large-mnli`
- **Purpose**: Zero-shot document classification
- **Accuracy**: High

**Alternatives:**
```env
# Faster classification
CLASSIFICATION_MODEL=facebook/bart-base

# Multilingual classification
CLASSIFICATION_MODEL=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

### LANGUAGE_DETECTION_MODEL
```env
LANGUAGE_DETECTION_MODEL=papluca/xlm-roberta-base-language-detection
```
- **Default**: `papluca/xlm-roberta-base-language-detection`
- **Purpose**: Detect input text language
- **Languages supported**: 20+ major languages

### TRANSLATION_MODEL
```env
TRANSLATION_MODEL=Helsinki-NLP/opus-mt-mul-en
```
- **Default**: `Helsinki-NLP/opus-mt-mul-en`
- **Purpose**: Translate between languages
- **Note**: System automatically selects specific models for language pairs

---

## Application Settings

### MAX_UPLOAD_SIZE_MB
```env
MAX_UPLOAD_SIZE_MB=50
```
- **Default**: 50
- **Range**: 1-100
- **Description**: Maximum file size for uploads
- **Recommendation**: 
  - Small docs: 10-20 MB
  - Large PDFs: 50-100 MB

### CHUNK_SIZE
```env
CHUNK_SIZE=500
```
- **Default**: 500 words
- **Range**: 200-1000
- **Description**: Size of text chunks for embedding
- **Impact**:
  - Smaller (200-300): Better for precise queries, more chunks
  - Medium (400-600): Balanced performance
  - Larger (700-1000): Better context, fewer chunks

**Recommendations by document type:**
```env
# Technical documentation
CHUNK_SIZE=300

# Narratives/books
CHUNK_SIZE=800

# Research papers
CHUNK_SIZE=500

# Legal documents
CHUNK_SIZE=600
```

### CHUNK_OVERLAP
```env
CHUNK_OVERLAP=50
```
- **Default**: 50 words
- **Range**: 0-200
- **Description**: Overlap between consecutive chunks
- **Purpose**: Preserve context across chunk boundaries
- **Formula**: Usually 10-20% of CHUNK_SIZE

**Recommendations:**
```env
# Minimal overlap (faster, less accurate)
CHUNK_OVERLAP=20

# Standard overlap
CHUNK_OVERLAP=50

# High overlap (slower, more accurate)
CHUNK_OVERLAP=100
```

### TOP_K_RETRIEVAL
```env
TOP_K_RETRIEVAL=3
```
- **Default**: 3
- **Range**: 1-10
- **Description**: Number of relevant chunks to retrieve per query
- **Impact**:
  - Lower (1-2): Faster, focused answers
  - Medium (3-5): Balanced
  - Higher (6-10): More context, slower

**Recommendations by use case:**
```env
# Quick facts
TOP_K_RETRIEVAL=1

# Standard Q&A
TOP_K_RETRIEVAL=3

# Complex analysis
TOP_K_RETRIEVAL=5

# Comprehensive research
TOP_K_RETRIEVAL=8
```

---

## Performance Optimization

### For Speed (Faster Responses)

```env
# Smaller, faster models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base

# Reduced retrieval
TOP_K_RETRIEVAL=2
CHUNK_SIZE=400
CHUNK_OVERLAP=40
```

### For Accuracy (Better Results)

```env
# Larger, more accurate models
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_MODEL=tiiuae/falcon-7b-instruct

# Increased context
TOP_K_RETRIEVAL=5
CHUNK_SIZE=600
CHUNK_OVERLAP=100
```

### For Multilingual Support

```env
# Best multilingual models
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=tiiuae/falcon-7b-instruct
CLASSIFICATION_MODEL=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

### For OCR-Heavy Workloads

```env
# Better OCR accuracy
OCR_MODEL=microsoft/trocr-large-printed

# Adjust chunk size for OCR text
CHUNK_SIZE=400
CHUNK_OVERLAP=60
```

---

## Use Case Configurations

### Configuration 1: Resume Screening System

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
CHUNK_SIZE=300
CHUNK_OVERLAP=30
TOP_K_RETRIEVAL=2
```

**Why:**
- Resumes are short documents
- Need quick processing
- Focused queries (skills, experience)

### Configuration 2: Research Paper Analysis

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_MODEL=tiiuae/falcon-7b-instruct
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=5
```

**Why:**
- Complex technical content
- Need comprehensive context
- Accuracy over speed

### Configuration 3: Multilingual Legal Documents

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.1
CLASSIFICATION_MODEL=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
CHUNK_SIZE=700
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=4
```

**Why:**
- Legal language requires context
- Multilingual support essential
- Higher accuracy needed

### Configuration 4: Scanned Document Processing

```env
OCR_MODEL=microsoft/trocr-large-printed
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHUNK_SIZE=400
CHUNK_OVERLAP=60
TOP_K_RETRIEVAL=3
```

**Why:**
- OCR quality is critical
- Medium chunk size for OCR text
- Standard retrieval settings

---

## API Rate Limits & Optimization

### Hugging Face Free Tier

- **Rate limit**: ~1000 requests/day per model
- **Concurrent requests**: Limited
- **Response time**: Can be slow during peak hours

### Optimization Strategies

1. **Reduce API Calls**
   ```env
   TOP_K_RETRIEVAL=2  # Fewer chunks = fewer processing steps
   ```

2. **Cache Results**
   - System caches embeddings during session
   - Reuse processed documents

3. **Batch Processing**
   - Upload multiple documents at once
   - Process together for efficiency

4. **Off-Peak Usage**
   - Use during off-peak hours for faster responses
   - US evening = Europe morning (less traffic)

### Upgrading Options

**Hugging Face Pro** ($9/month):
- Higher rate limits
- Faster inference
- Priority access
- Better for production

**Enterprise Options**:
- Dedicated endpoints
- SLA guarantees
- Custom models
- Contact Hugging Face sales

---

## Debugging Configuration

### Enable Detailed Logging

In `app.py`, change:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Test Individual Components

```bash
# Test OCR
python -c "from pipeline import DocumentProcessor; print('OCR model available')"

# Test embeddings
python -c "from rag_engine import MultilingualRAGEngine; print('Embeddings working')"

# Test API connection
python utils.py --test-api
```

### Monitor Performance

Add timing logs to key functions:
```python
import time
start = time.time()
# your code
print(f"Took {time.time() - start:.2f} seconds")
```

---

## Security Best Practices

### API Key Protection

1. **Never commit .env to git**
   ```bash
   # Already in .gitignore
   .env
   ```

2. **Use environment variables in production**
   ```bash
   # Set system environment variable
   setx HUGGINGFACE_API_KEY "hf_xxx"  # Windows
   export HUGGINGFACE_API_KEY="hf_xxx"  # Linux/Mac
   ```

3. **Rotate keys regularly**
   - Generate new tokens every 90 days
   - Revoke old tokens

### Document Privacy

1. **PII Masking**
   - System masks email, phone, SSN
   - Review advanced_features.py for patterns

2. **Local Processing**
   - Consider local models for sensitive docs
   - See future roadmap for local deployment

3. **Temporary Storage**
   - Clear uploads/ folder regularly
   - Don't store sensitive documents long-term

---

## Troubleshooting Configuration Issues

### Issue: Models not loading

**Check:**
```bash
python utils.py --test-api
```

**Solution:**
- Verify API key is correct
- Check internet connection
- Try different model

### Issue: Out of memory

**Solution:**
```env
CHUNK_SIZE=300
TOP_K_RETRIEVAL=2
```

### Issue: Slow responses

**Solution:**
```env
# Use faster models
LLM_MODEL=google/flan-t5-base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Issue: Poor accuracy

**Solution:**
```env
# Use better models
TOP_K_RETRIEVAL=5
CHUNK_OVERLAP=100
```

---

## Getting Help

- **Documentation**: README.md
- **Quick Start**: QUICKSTART.md
- **Diagnostics**: `python utils.py --diagnose`
- **Examples**: `python utils.py --examples`

---

**Last Updated**: 2024
**Version**: 1.0
**Maintained by**: OmniDoc AI Team
