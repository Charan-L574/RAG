"""
Utility functions and helpers for OmniDoc AI
"""

import os
from pathlib import Path
from typing import List, Dict
import json


def create_sample_env():
    """Create a sample .env file if it doesn't exist"""
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write("""# Hugging Face API Configuration
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=tiiuae/falcon-7b-instruct
OCR_MODEL=microsoft/trocr-base-printed
CLASSIFICATION_MODEL=facebook/bart-large-mnli
LANGUAGE_DETECTION_MODEL=papluca/xlm-roberta-base-language-detection
TRANSLATION_MODEL=Helsinki-NLP/opus-mt-mul-en

# Application Settings
MAX_UPLOAD_SIZE_MB=50
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=3
""")
        print("✅ Created .env file. Please add your Hugging Face API key.")
    else:
        print("ℹ️ .env file already exists.")


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'langchain',
        'transformers',
        'gradio',
        'faiss',
        'PyPDF2',
        'python-docx',
        'pillow',
        'pandas',
        'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed!")
        return True


def validate_api_key(api_key: str) -> bool:
    """Validate Hugging Face API key format"""
    if not api_key or api_key == "your_huggingface_api_key_here":
        print("❌ Invalid API key. Please add your Hugging Face API key to .env file")
        print("   Get your key from: https://huggingface.co/settings/tokens")
        return False
    
    if not api_key.startswith('hf_'):
        print("⚠️ Warning: API key should start with 'hf_'")
        return False
    
    print("✅ API key format looks valid")
    return True


def create_test_documents():
    """Create sample test documents for testing"""
    test_dir = Path('test_documents')
    test_dir.mkdir(exist_ok=True)
    
    # Sample resume
    resume_content = """
    JOHN DOE
    Software Engineer
    Email: john.doe@example.com | Phone: (555) 123-4567
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    Expertise in Python, JavaScript, and cloud technologies.
    
    WORK EXPERIENCE
    
    Senior Software Engineer | Tech Corp | 2020-Present
    - Led development of microservices architecture
    - Improved system performance by 40%
    - Mentored junior developers
    
    Software Engineer | StartupXYZ | 2018-2020
    - Built RESTful APIs using Python and Flask
    - Implemented CI/CD pipelines
    - Worked with AWS services
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2014-2018
    
    SKILLS
    - Programming: Python, JavaScript, Java, SQL
    - Frameworks: Django, React, Flask, FastAPI
    - Cloud: AWS, Azure, Docker, Kubernetes
    - Tools: Git, Jenkins, Jira
    """
    
    with open(test_dir / 'sample_resume.txt', 'w') as f:
        f.write(resume_content)
    
    # Sample research abstract
    research_content = """
    TITLE: A Novel Approach to Multilingual Text Processing
    
    ABSTRACT
    This paper presents a novel approach to multilingual text processing using
    transformer-based models. We demonstrate that cross-lingual embeddings can
    significantly improve the performance of downstream tasks across multiple
    languages.
    
    INTRODUCTION
    Multilingual natural language processing has gained significant attention in
    recent years. However, many existing approaches require language-specific
    models or extensive parallel corpora.
    
    METHODOLOGY
    We employed a transformer-based architecture with multilingual pre-training.
    Our model was trained on a corpus of 100 languages using masked language
    modeling and translation language modeling objectives.
    
    RESULTS
    Our experiments show that the proposed approach achieves state-of-the-art
    results on several benchmark datasets. The model demonstrates strong
    zero-shot cross-lingual transfer capabilities.
    
    CONCLUSION
    This work contributes to the field of multilingual NLP by providing an
    effective and efficient approach to cross-lingual understanding. Future
    work will explore applications in low-resource languages.
    """
    
    with open(test_dir / 'sample_research.txt', 'w') as f:
        f.write(research_content)
    
    # Sample invoice
    invoice_content = """
    INVOICE
    
    Invoice Number: INV-2024-001
    Date: January 15, 2024
    Due Date: February 15, 2024
    
    From:
    ABC Services Inc.
    123 Business Street
    New York, NY 10001
    
    To:
    XYZ Corporation
    456 Corporate Ave
    Boston, MA 02101
    
    ITEMS:
    1. Consulting Services (40 hours @ $150/hr)    $6,000.00
    2. Software Development (80 hours @ $120/hr)   $9,600.00
    3. Project Management (20 hours @ $100/hr)     $2,000.00
    
    Subtotal:                                      $17,600.00
    Tax (8%):                                      $1,408.00
    
    TOTAL:                                         $19,008.00
    
    Payment Terms: Net 30 days
    Payment Method: Bank transfer or check
    """
    
    with open(test_dir / 'sample_invoice.txt', 'w') as f:
        f.write(invoice_content)
    
    print(f"✅ Created test documents in {test_dir}/")
    return test_dir


def run_diagnostics():
    """Run system diagnostics"""
    print("🔍 Running OmniDoc AI Diagnostics...\n")
    
    print("1. Checking Python version...")
    import sys
    print(f"   Python {sys.version}")
    
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n3. Checking environment file...")
    if not Path('.env').exists():
        print("   ⚠️ .env file not found")
        create_sample_env()
    else:
        print("   ✅ .env file exists")
    
    print("\n4. Checking API key...")
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('HUGGINGFACE_API_KEY', '')
    validate_api_key(api_key)
    
    print("\n5. Checking test documents...")
    if not Path('test_documents').exists():
        print("   Creating test documents...")
        create_test_documents()
    else:
        print("   ✅ Test documents directory exists")
    
    print("\n" + "="*60)
    if deps_ok and Path('.env').exists() and api_key:
        print("✅ System ready! Run: python app.py")
    else:
        print("⚠️ Please fix the issues above before running the app")
    print("="*60)


def test_huggingface_api(api_key: str):
    """Test Hugging Face API connection"""
    import requests
    
    print("🧪 Testing Hugging Face API connection...")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    test_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        response = requests.post(
            test_url,
            headers=headers,
            json={"inputs": "Hello, world!"},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API connection successful!")
            return True
        elif response.status_code == 401:
            print("❌ Authentication failed. Check your API key.")
            return False
        else:
            print(f"⚠️ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def print_usage_examples():
    """Print usage examples"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              OmniDoc AI - Usage Examples                       ║
╚════════════════════════════════════════════════════════════════╝

📄 RESUME ANALYSIS
  Upload: resume.pdf
  Questions:
    - "What are the candidate's top skills?"
    - "How many years of experience in Python?"
    - "Summarize the education background"

📚 RESEARCH PAPER
  Upload: paper.pdf
  Questions:
    - "What is the main research question?"
    - "Explain the methodology"
    - "What are the key findings?"

🌐 MULTILINGUAL QUERIES
  Upload: document.pdf (English)
  Questions:
    - In Hindi: "इस दस्तावेज़ का सारांश दें"
    - In Spanish: "¿Cuáles son los puntos principales?"
    - In Tamil: "இந்த ஆவணத்தின் சுருக்கம் என்ன?"

📖 SCANNED DOCUMENTS
  Upload: scanned_textbook.jpg
  System automatically performs OCR
  Ask: "What topics are covered in this page?"

💰 INVOICE PROCESSING
  Upload: invoice.pdf
  Questions:
    - "What is the total amount?"
    - "Who is the vendor?"
    - "When is payment due?"

📊 MULTIPLE DOCUMENTS
  Upload: doc1.pdf, doc2.pdf, doc3.pdf
  Questions:
    - "Compare the main themes across all documents"
    - "Which document discusses topic X?"
    - "Summarize each document"

═══════════════════════════════════════════════════════════════

🚀 Quick Start:
  1. python utils.py --diagnose    # Check system
  2. python app.py                 # Start application
  3. Open http://localhost:7860    # Access UI

═══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--diagnose":
            run_diagnostics()
        elif sys.argv[1] == "--test-api":
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('HUGGINGFACE_API_KEY', '')
            if api_key:
                test_huggingface_api(api_key)
            else:
                print("❌ No API key found in .env file")
        elif sys.argv[1] == "--create-test-docs":
            create_test_documents()
        elif sys.argv[1] == "--examples":
            print_usage_examples()
        else:
            print("""
Usage:
  python utils.py --diagnose          # Run system diagnostics
  python utils.py --test-api          # Test Hugging Face API
  python utils.py --create-test-docs  # Create sample documents
  python utils.py --examples          # Show usage examples
""")
    else:
        print_usage_examples()
