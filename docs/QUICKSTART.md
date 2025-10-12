# 🚀 Quick Start Guide - OmniDoc AI

Get up and running with OmniDoc AI in 5 minutes!

---

## ⚡ Installation (Choose One Method)

### Method 1: Automated Setup (Recommended for Beginners)

```bash
# Run the setup wizard
python setup.py
```

The wizard will:
- Check Python version
- Create virtual environment
- Install all dependencies
- Set up environment variables
- Create necessary directories

### Method 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure .env
copy .env.example .env
# Edit .env and add your Hugging Face API key
```

---

## 🔑 Get Your Hugging Face API Key

1. Go to [Hugging Face](https://huggingface.co/)
2. Create a free account (if you don't have one)
3. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Click "New token"
5. Name it "OmniDoc AI" and select "Read" permission
6. Copy the token
7. Paste it in your `.env` file:
   ```
   HUGGINGFACE_API_KEY=hf_your_token_here
   ```

---

## ▶️ Run the Application

```bash
# Make sure virtual environment is activated
python app.py
```

Open your browser and go to: **http://localhost:7860**

---

## 📖 First Steps Tutorial

### Step 1: Upload a Document

1. Click **"Upload Files"** button
2. Select one or more files:
   - PDF, DOCX, TXT, CSV, XLSX, PPTX
   - Images: JPG, PNG (with OCR)
3. Click **"Process Documents"**
4. Wait for processing (usually 10-30 seconds)

### Step 2: Review Auto-Insights

After processing, you'll see:
- ✅ **Processing Status** - Document type and language detected
- 📊 **Document Information** - Stats and metadata
- 💡 **Suggested Questions** - Smart questions to ask
- 🔍 **Auto-Generated Insights** - Summary and key points

### Step 3: Ask Questions

1. Type your question in the text box
2. Questions can be in **any language**!
3. Click **"Send"** or press Enter
4. View the answer with source citations

### Step 4: Follow-Up Questions

The system remembers context! You can ask:
- "Tell me more about that"
- "Can you explain it differently?"
- "What about the second point?"

---

## 💡 Example Workflows

### 📄 Resume Analysis

```
1. Upload: resume.pdf
2. Check suggested questions
3. Ask: "What are the candidate's top 5 skills?"
4. Ask: "Summarize work experience"
5. Ask: "What is the education level?"
```

### 📚 Research Paper

```
1. Upload: research_paper.pdf
2. Auto-generated insights show summary
3. Ask: "What is the research question?"
4. Ask: "Explain the methodology in simple terms"
5. Ask: "What are the main findings?"
```

### 🌐 Multilingual Document

```
1. Upload: english_document.pdf
2. Ask in Hindi: "मुख्य बिंदु क्या हैं?"
3. Get answer in Hindi!
4. Switch languages anytime
```

### 📖 Scanned Textbook

```
1. Upload: textbook_page.jpg
2. System performs automatic OCR
3. Ask: "What topics are covered?"
4. Ask: "Explain the first example"
```

---

## 🎯 Tips for Best Results

### ✅ Do's

- **Use specific questions** - "What is the total amount in the invoice?" is better than "Tell me about the invoice"
- **Upload clear images** - Higher quality = better OCR results
- **Try suggested questions** - They're tailored to your document type
- **Enable translation** - For multilingual workflows
- **Ask follow-up questions** - System remembers context

### ❌ Don'ts

- **Don't upload huge files** - Keep PDFs under 50MB
- **Don't ask about non-existent content** - System will tell you if information isn't available
- **Don't expect instant responses** - Free API can take 5-10 seconds
- **Don't upload sensitive documents** - Without proper security measures

---

## 🔧 Common Issues & Solutions

### Issue: "API key not found"

**Solution:**
```bash
# Check if .env file exists
dir .env  # Windows
ls .env   # Linux/Mac

# If not, create it from example
copy .env.example .env

# Edit and add your API key
```

### Issue: Slow responses

**Solution:**
- Free tier has rate limits
- Reduce document size
- Lower `TOP_K_RETRIEVAL` in .env
- Consider upgrading to HF Pro

### Issue: OCR not working

**Solution:**
- Ensure image is clear and high-resolution
- Try converting PDF to images first
- Check if OCR model is accessible in your region

### Issue: Translation not working

**Solution:**
- Some language pairs may not be supported
- System will fallback to English
- Check specific translation model availability

---

## 🧪 Test Your Setup

```bash
# Run diagnostics
python utils.py --diagnose

# Test API connection
python utils.py --test-api

# Create sample documents
python utils.py --create-test-docs

# View usage examples
python utils.py --examples
```

---

## 📚 Advanced Features

### Enable/Disable Translation

Toggle the **"Enable multilingual translation"** checkbox:
- ✅ **On**: Ask in any language, get answers translated
- ❌ **Off**: Faster responses, English only

### View Source Context

Check the **"Sources & Context"** section to see:
- Which documents were used
- Relevant page numbers
- Similarity scores
- Content excerpts

### Clear Conversation

Click **"Clear Conversation"** to:
- Reset chat history
- Start fresh context
- Maintain uploaded documents

---

## 🎓 Learning Path

### Beginner
1. Upload a simple TXT file
2. Ask basic questions
3. Explore suggested questions
4. View auto-generated insights

### Intermediate
1. Upload PDFs with multiple pages
2. Ask complex questions
3. Try follow-up questions
4. Test multilingual queries

### Advanced
1. Upload multiple documents
2. Compare documents
3. Use scanned images with OCR
4. Experiment with different languages
5. Adjust settings in .env file

---

## 📞 Getting Help

### Documentation
- **README.md** - Full documentation
- **Code comments** - Detailed explanations
- **.env.example** - Configuration guide

### Troubleshooting
1. Run: `python utils.py --diagnose`
2. Check error messages in terminal
3. Verify API key is correct
4. Ensure all dependencies installed

### Community
- Open GitHub issue for bugs
- Check existing issues for solutions
- Review Hugging Face docs for API limits

---

## 🎉 You're Ready!

Start exploring OmniDoc AI:

```bash
python app.py
```

### Try These First:
1. Upload `test_documents/sample_resume.txt`
2. Click suggested questions
3. Ask: "What are the key skills?"
4. Try asking in your native language!

---

## 🔄 Updating

To get latest updates:

```bash
# Pull latest code
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart application
python app.py
```

---

## 📊 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- Internet connection (for API calls)
- Modern web browser

**Recommended:**
- Python 3.10+
- 8GB RAM
- Fast internet connection
- Chrome/Firefox browser

---

**Happy document chatting! 🚀**

*For detailed information, see README.md*
