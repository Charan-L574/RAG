# ✅ Installation & Verification Guide

Step-by-step guide to install, configure, and verify OmniDoc AI is working correctly.

---

## 📋 Pre-Installation Checklist

Before starting, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] Internet connection (for API calls)
- [ ] Text editor (VS Code, Notepad++, etc.)
- [ ] Terminal/Command Prompt access
- [ ] Hugging Face account (free - sign up at https://huggingface.co/)

---

## 🚀 Installation Methods

Choose one of these methods:

### Method 1: Automated Setup (Recommended for Beginners) ⭐

This is the easiest way to get started.

```powershell
# Step 1: Navigate to the project folder
cd C:\Users\Charan\Desktop\rag

# Step 2: Run the setup wizard
python setup.py
```

The wizard will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set up configuration files
- ✅ Create necessary directories

Follow the prompts and provide your Hugging Face API key when asked.

---

### Method 2: Manual Setup (For Advanced Users)

If you prefer manual control:

```powershell
# Step 1: Navigate to project folder
cd C:\Users\Charan\Desktop\rag

# Step 2: Create virtual environment
python -m venv venv

# Step 3: Activate virtual environment
# On Windows PowerShell:
venv\Scripts\activate
# On Windows CMD:
venv\Scripts\activate.bat

# Step 4: Upgrade pip
python -m pip install --upgrade pip

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: Create .env file
copy .env.example .env

# Step 7: Edit .env file
# Use notepad, VS Code, or any text editor
notepad .env
```

---

## 🔑 Getting Your Hugging Face API Key

### Step-by-Step Guide

1. **Visit Hugging Face**
   - Go to: https://huggingface.co/

2. **Sign Up (if new user)**
   - Click "Sign Up" in the top right
   - Use email or Google/GitHub login
   - Verify your email
   - **It's completely FREE!**

3. **Go to Settings**
   - Click your profile picture (top right)
   - Select "Settings"
   - Click "Access Tokens" in the left menu

4. **Create New Token**
   - Click "New token"
   - Give it a name: "OmniDoc AI"
   - Select role: "Read"
   - Click "Generate token"

5. **Copy Your Token**
   - Your token starts with `hf_`
   - Click the copy button
   - **Keep it secret!** (Don't share publicly)

6. **Add to .env File**
   ```env
   HUGGINGFACE_API_KEY=hf_your_copied_token_here
   ```

---

## ⚙️ Configuration

### Basic Configuration (Recommended)

Your `.env` file should look like this:

```env
# Required: Add your actual API key here
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxx

# Models (defaults are fine for most users)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=tiiuae/falcon-7b-instruct
OCR_MODEL=microsoft/trocr-base-printed
CLASSIFICATION_MODEL=facebook/bart-large-mnli
LANGUAGE_DETECTION_MODEL=papluca/xlm-roberta-base-language-detection
TRANSLATION_MODEL=Helsinki-NLP/opus-mt-mul-en

# Settings (defaults are fine for beginners)
MAX_UPLOAD_SIZE_MB=50
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=3
```

**Important**: Only change `HUGGINGFACE_API_KEY`. Leave other settings as default for now.

---

## ✅ Verification Steps

### Step 1: Verify Python Installation

```powershell
python --version
```

**Expected Output**: `Python 3.8.x` or higher

**If error**: Python not installed. Download from https://www.python.org/

---

### Step 2: Verify Virtual Environment

```powershell
# Activate venv first
venv\Scripts\activate

# Your prompt should change to show (venv)
# Example: (venv) PS C:\Users\Charan\Desktop\rag>
```

**Expected**: Prompt shows `(venv)` prefix

**If error**: 
```powershell
# Try creating venv again
python -m venv venv
```

---

### Step 3: Verify Dependencies

```powershell
# Run diagnostics
python utils.py --diagnose
```

**Expected Output**:
```
🔍 Running OmniDoc AI Diagnostics...

1. Checking Python version...
   Python 3.x.x

2. Checking dependencies...
   ✅ All required packages are installed!

3. Checking environment file...
   ✅ .env file exists

4. Checking API key...
   ✅ API key format looks valid

5. Checking test documents...
   ✅ Test documents directory exists

════════════════════════════════════════════════
✅ System ready! Run: python app.py
════════════════════════════════════════════════
```

**If any ❌ appears**: Follow the suggestions provided

---

### Step 4: Test API Connection

```powershell
python utils.py --test-api
```

**Expected Output**:
```
🧪 Testing Hugging Face API connection...
✅ API connection successful!
```

**If error "Authentication failed"**:
- Check your API key in `.env`
- Ensure it starts with `hf_`
- Try generating a new token

**If error "Connection error"**:
- Check internet connection
- Try again (API might be temporarily busy)

---

### Step 5: Create Test Documents

```powershell
python utils.py --create-test-docs
```

**Expected Output**:
```
✅ Created test documents in test_documents/
```

**Verify**: Check `test_documents/` folder exists with 3 files:
- `sample_resume.txt`
- `sample_research.txt`
- `sample_invoice.txt`

---

### Step 6: Run the Application

```powershell
python app.py
```

**Expected Output**:
```
INFO:__main__:Initializing OmniDoc AI components...
INFO:__main__:OmniDoc AI initialized successfully!
INFO:__main__:Launching OmniDoc AI...
Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```

**If error "Port 7860 already in use"**:
```powershell
# Stop other instance or use different port
# Edit app.py and change: server_port=7861
```

---

### Step 7: Test in Browser

1. **Open Browser**
   - Chrome, Firefox, Edge, or Safari
   - Navigate to: http://localhost:7860

2. **Expected Screen**:
   - Title: "🌍 OmniDoc AI"
   - Subtitle: "Multilingual Intelligent Document Conversational Assistant"
   - Upload panel on left
   - Chat interface on right

3. **If blank screen**:
   - Wait 30 seconds (first load can be slow)
   - Refresh browser (F5)
   - Check terminal for errors

---

### Step 8: Test Document Upload

1. **Click "Upload Files"**
2. **Select**: `test_documents/sample_resume.txt`
3. **Click**: "🔄 Process Documents"
4. **Wait**: 10-30 seconds
5. **Expected**:
   - ✅ Status shows: "sample_resume.txt - Resume or CV"
   - Document info appears
   - Suggested questions appear
   - Insights appear

**If error**:
- Check file is not too large
- Verify API key is correct
- Check internet connection

---

### Step 9: Test Question Answering

1. **In question box, type**: "What are the top skills?"
2. **Click**: "🚀 Send"
3. **Wait**: 5-15 seconds
4. **Expected**:
   - Answer appears in chat
   - Sources shown below
   - Context displayed

**If slow**:
- Free API tier can be slow
- Normal wait time: 10-30 seconds
- Try during off-peak hours

---

### Step 10: Test Multilingual Support

1. **Type in Hindi**: "मुख्य कौशल क्या हैं?"
2. **OR in Spanish**: "¿Cuáles son las principales habilidades?"
3. **Click Send**
4. **Expected**:
   - Answer in same language as question
   - Translation enabled by default

**If not translating**:
- Check "Enable multilingual translation" checkbox is enabled
- Some language pairs may not be supported
- System will fallback to English

---

## 🎯 Common Issues & Solutions

### Issue 1: "Module not found" error

**Solution**:
```powershell
# Ensure venv is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Issue 2: "API key not found"

**Solution**:
```powershell
# Check .env file exists
dir .env

# Verify content
notepad .env

# Ensure format is correct:
# HUGGINGFACE_API_KEY=hf_your_key_here
# (no spaces around =)
```

---

### Issue 3: "Connection timeout" or slow responses

**Cause**: Free API tier has rate limits

**Solutions**:
- Wait a few minutes between requests
- Try during off-peak hours (US evening)
- Reduce `TOP_K_RETRIEVAL` in .env to 2
- Consider upgrading to HF Pro ($9/month)

---

### Issue 4: OCR not working

**Solutions**:
- Ensure image is clear and high-resolution
- Try with `.jpg` or `.png` format
- Check if OCR model is available in your region
- Try alternative OCR model in .env

---

### Issue 5: Out of memory

**Solutions**:
```env
# In .env file, reduce these values:
CHUNK_SIZE=300
TOP_K_RETRIEVAL=2
```

---

### Issue 6: PowerShell execution policy error

**Error**: "cannot be loaded because running scripts is disabled"

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating venv again
venv\Scripts\activate
```

---

## 🧪 Testing Checklist

After installation, verify each feature:

### Core Features
- [ ] Document upload (PDF, DOCX, TXT, images)
- [ ] Text extraction
- [ ] Document classification
- [ ] Language detection
- [ ] Question answering
- [ ] Source citations

### Advanced Features
- [ ] Auto-generated questions
- [ ] Document insights
- [ ] Multilingual queries
- [ ] Follow-up questions
- [ ] Multiple document upload

### UI Features
- [ ] File upload panel
- [ ] Chat interface
- [ ] Suggested questions panel
- [ ] Insights panel
- [ ] Sources display
- [ ] Clear conversation button

---

## 📊 Performance Benchmarks

What to expect on free tier:

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Document upload | 1-5 sec | Depends on size |
| Document processing | 10-30 sec | First time slower |
| Question answering | 5-15 sec | API dependent |
| Language detection | 1-2 sec | Fast |
| Translation | 3-10 sec | Language dependent |
| OCR (per page) | 15-30 sec | Image quality matters |

**Tips for better performance**:
- Use during off-peak hours
- Process smaller documents
- Reduce TOP_K_RETRIEVAL
- Cache frequently used docs

---

## 🎓 Next Steps After Installation

### For Beginners
1. ✅ Upload `sample_resume.txt`
2. ✅ Click suggested questions
3. ✅ Try asking your own questions
4. ✅ Upload your own document
5. ✅ Explore other features

### For Intermediate Users
1. ✅ Try multilingual queries
2. ✅ Upload multiple documents
3. ✅ Test with scanned images
4. ✅ Adjust settings in .env
5. ✅ Read CONFIGURATION.md

### For Advanced Users
1. ✅ Review source code
2. ✅ Customize prompts
3. ✅ Add new features
4. ✅ Optimize performance
5. ✅ Deploy to production

---

## 📚 Additional Resources

### Documentation
- **README.md**: Full documentation
- **QUICKSTART.md**: Quick start guide
- **CONFIGURATION.md**: Configuration options
- **ARCHITECTURE.md**: System architecture

### Commands Reference
```powershell
# Diagnostics
python utils.py --diagnose

# Test API
python utils.py --test-api

# Create sample docs
python utils.py --create-test-docs

# View examples
python utils.py --examples

# Run application
python app.py
```

### Useful Links
- Hugging Face: https://huggingface.co/
- LangChain Docs: https://python.langchain.com/
- Gradio Docs: https://gradio.app/
- Project GitHub: [Your repository URL]

---

## 🆘 Getting Help

If you're stuck:

1. **Run Diagnostics**
   ```powershell
   python utils.py --diagnose
   ```

2. **Check Error Messages**
   - Read terminal output carefully
   - Most errors include solutions

3. **Review Documentation**
   - README.md for features
   - QUICKSTART.md for basics
   - CONFIGURATION.md for settings

4. **Common Solutions**
   - Restart application
   - Check internet connection
   - Verify API key
   - Update dependencies
   - Check .env format

5. **Still Stuck?**
   - Create GitHub issue
   - Include error message
   - Share diagnostics output
   - Describe what you tried

---

## ✅ Installation Complete!

If you've reached here and all tests passed:

🎉 **Congratulations!** OmniDoc AI is ready to use!

### Quick Summary
- ✅ Python installed
- ✅ Dependencies installed
- ✅ API key configured
- ✅ Application tested
- ✅ All features working

### Start Using
```powershell
# Activate venv
venv\Scripts\activate

# Run app
python app.py

# Open browser
# http://localhost:7860
```

### Enjoy!
- Upload any document
- Ask questions in any language
- Explore all features
- Customize as needed

---

**Happy document chatting! 🚀**

*For questions and support, refer to the documentation or create an issue.*

---

**Installation Guide Version**: 1.0
**Last Updated**: 2024
**Status**: Production Ready ✅
