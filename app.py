"""
OmniDoc AI - Multilingual Intelligent Document Conversational Assistant
Main Gradio Application
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import gradio as gr

from pipeline import DocumentProcessor, DocumentClassifier
from rag_engine import MultilingualRAGEngine
from multilingual import MultilingualProcessor, ContextAwarePromptBuilder
from advanced_features import (
    QuestionGenerator,
    PIIMasker,
    DocumentInsights,
    ConversationMemory,
    DocumentComparator
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OmniDocAI:
    """Main application class"""
    
    def __init__(self):
        # Load configuration
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not self.hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        
        # Model configurations
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.llm_model = os.getenv('LLM_MODEL', 'tiiuae/falcon-7b-instruct')
        self.ocr_model = os.getenv('OCR_MODEL', 'microsoft/trocr-base-printed')
        self.classification_model = os.getenv('CLASSIFICATION_MODEL', 'facebook/bart-large-mnli')
        self.language_detection_model = os.getenv('LANGUAGE_DETECTION_MODEL', 'papluca/xlm-roberta-base-language-detection')
        self.translation_model = os.getenv('TRANSLATION_MODEL', 'Helsinki-NLP/opus-mt-mul-en')
        
        # Application settings
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        self.top_k = int(os.getenv('TOP_K_RETRIEVAL', 3))
        
        # Initialize components
        logger.info("Initializing OmniDoc AI components...")
        
        self.doc_processor = DocumentProcessor(
            self.hf_api_key,
            self.ocr_model,
            self.classification_model
        )
        
        self.rag_engine = MultilingualRAGEngine(
            self.hf_api_key,
            self.embedding_model,
            self.llm_model,
            self.chunk_size,
            self.chunk_overlap,
            self.top_k
        )
        
        self.multilingual_processor = MultilingualProcessor(
            self.hf_api_key,
            self.language_detection_model,
            self.translation_model
        )
        
        self.question_generator = QuestionGenerator(
            self.hf_api_key,
            self.llm_model
        )
        
        self.insights_generator = DocumentInsights(
            self.hf_api_key,
            self.llm_model
        )
        
        self.conversation_memory = ConversationMemory(max_history=10)
        
        # State
        self.processed_documents = []
        self.current_document_type = "Generic Document"
        
        logger.info("OmniDoc AI initialized successfully!")
    
    def process_uploaded_files(self, files) -> Tuple[str, str, str, str, str]:
        """
        Process uploaded files
        
        Returns:
            Tuple of (status_message, document_info, suggested_questions, insights, detected_doc_type)
        """
        if not files:
            return "No files uploaded", "", "", "", "Generic Document"
        
        try:
            # Clear previous documents
            self.processed_documents = []
            self.rag_engine.clear()
            self.conversation_memory.clear()
            
            status_messages = []
            document_info_parts = []
            all_questions = []
            all_insights = []
            
            # Process each file
            for file in files:
                file_path = file.name if hasattr(file, 'name') else file
                logger.info(f"Processing file: {file_path}")
                
                # Process document
                processed_doc = self.doc_processor.process_document(file_path)
                self.processed_documents.append(processed_doc)
                
                # Extract info
                metadata = processed_doc['metadata']
                doc_type = processed_doc['document_type']
                extracted_text = processed_doc['extracted_text']
                
                # Detect document language
                lang_info = self.multilingual_processor.detect_language(extracted_text[:500])
                
                status_messages.append(f"✅ {metadata['filename']} - {doc_type}")
                
                # Document info
                info = f"""
📄 **{metadata['filename']}**
- Type: {doc_type}
- Language: {lang_info['name']} ({lang_info['code']})
- Pages: {metadata.get('total_pages', 1)}
- Words: {len(extracted_text.split())}
"""
                document_info_parts.append(info)
                
                # Generate suggested questions
                questions = self.question_generator.generate_questions(
                    extracted_text,
                    doc_type,
                    num_questions=5
                )
                
                if questions:
                    questions_text = f"\n**Questions for {metadata['filename']}:**\n"
                    for i, q in enumerate(questions, 1):
                        questions_text += f"{i}. {q}\n"
                    all_questions.append(questions_text)
                
                # Generate insights
                insights = self.insights_generator.generate_insights(
                    extracted_text,
                    doc_type,
                    metadata
                )
                
                insights_text = f"""
**Insights for {metadata['filename']}:**

📊 Statistics:
- Words: {insights['statistics']['word_count']}
- Sentences: {insights['statistics']['sentence_count']}
- Pages: {insights['statistics']['pages']}

📝 Summary:
{insights['summary']}

🔑 Key Points:
{chr(10).join(['- ' + kp for kp in insights['key_points'][:5]])}
"""
                all_insights.append(insights_text)
            
            # Add documents to RAG engine
            chunks_added = self.rag_engine.add_documents(self.processed_documents)
            
            # Set current document type (use first document's type)
            if self.processed_documents:
                self.current_document_type = self.processed_documents[0]['document_type']
            
            # Comparison if multiple documents
            comparison_text = ""
            if len(self.processed_documents) > 1:
                comparison = DocumentComparator.compare_documents(self.processed_documents)
                comparison_text = f"\n\n📊 **Document Comparison:**\n"
                comparison_text += f"- Total documents: {comparison['total_documents']}\n"
                comparison_text += f"- Document types: {', '.join([f'{k} ({v})' for k, v in comparison['document_types'].items()])}\n"
            
            status = "\n".join(status_messages)
            status += f"\n\n✨ Created {chunks_added} text chunks for RAG"
            status += comparison_text
            
            doc_info = "\n".join(document_info_parts)
            questions = "\n".join(all_questions)
            insights = "\n".join(all_insights)
            
            return status, doc_info, questions, insights, self.current_document_type
            
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return f"❌ Error: {str(e)}", "", "", "", "Generic Document"
    
    def answer_question(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
        enable_translation: bool = True
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Answer user question
        
        Returns:
            Tuple of (updated_chat_history, sources_text)
        """
        if not query.strip():
            return chat_history, ""
        
        if not self.processed_documents:
            response = "⚠️ Please upload documents first before asking questions."
            chat_history.append((query, response))
            return chat_history, ""
        
        try:
            # Detect query language
            query_lang = self.multilingual_processor.detect_language(query)
            logger.info(f"Query language detected: {query_lang['name']}")
            
            # Translate query if needed (to English for retrieval)
            search_query = query
            if enable_translation and query_lang['code'] != 'en':
                translated = self.multilingual_processor.translate_text(
                    query,
                    query_lang['code'],
                    'en'
                )
                if translated:
                    search_query = translated
                    logger.info(f"Query translated to: {search_query}")
            
            # Retrieve relevant chunks
            relevant_chunks = self.rag_engine.retrieve_relevant_chunks(search_query)
            
            if not relevant_chunks:
                response = "I couldn't find relevant information in the documents to answer your question."
                chat_history.append((query, response))
                return chat_history, ""
            
            # Generate answer with document type awareness
            answer = self.rag_engine.generate_answer(
                search_query,
                relevant_chunks,
                document_type=self.current_document_type,
                language=query_lang['code'] if enable_translation else 'en'
            )
            
            # Translate answer back if needed
            if enable_translation and query_lang['code'] != 'en':
                translated_answer = self.multilingual_processor.translate_response(
                    answer,
                    query_lang['code']
                )
                if translated_answer:
                    answer = translated_answer
            
            # Add to conversation memory
            self.conversation_memory.add_interaction(query, answer, relevant_chunks)
            
            # Build sources text
            sources_text = self._format_sources(relevant_chunks)
            
            # Update chat history
            chat_history.append((query, answer))
            
            return chat_history, sources_text
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            error_msg = f"❌ Error generating answer: {str(e)}"
            chat_history.append((query, error_msg))
            return chat_history, ""
    
    def update_document_type(self, doc_type: str):
        """Update the current document type when user changes it"""
        self.current_document_type = doc_type
        logger.info(f"Document type manually updated to: {doc_type}")
        return f"✅ Document type updated to: {doc_type}"
    
    def _format_sources(self, chunks: List[dict]) -> str:
        """Format source information"""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            source = f"""
**Source {i}:**
- Document: {metadata.get('filename', 'Unknown')}
- Page: {metadata.get('page_number', 'N/A')}
- Relevance: {chunk.get('score', 0):.2%}

*Content excerpt:*
{chunk['content'][:200]}...
"""
            sources.append(source)
        
        return "\n---\n".join(sources)
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_memory.clear()
        return [], ""


def create_ui(app: OmniDocAI):
    """Create Gradio UI"""
    
    with gr.Blocks(
        title="OmniDoc AI",
        theme=gr.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .feature-box {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🌍 OmniDoc AI</h1>
            <p><strong>Multilingual Intelligent Document Conversational Assistant</strong></p>
            <p>Upload documents, ask questions in any language, get intelligent answers with OCR support</p>
        </div>
        """)
        
        # Main layout
        with gr.Row():
            # Left column - Document Upload
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload Documents")
                
                file_upload = gr.File(
                    label="Upload Files",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".csv", ".xlsx", ".pptx", ".jpg", ".jpeg", ".png"]
                )
                
                process_btn = gr.Button("🔄 Process Documents", variant="primary", size="lg")
                
                gr.Markdown("### 📝 Document Type")
                doc_type_dropdown = gr.Dropdown(
                    choices=[
                        "Job Description",
                        "Resume or CV",
                        "Research Paper or Academic Article",
                        "Legal Document or Contract",
                        "Invoice or Financial Report",
                        "Textbook or Educational Material",
                        "Generic Document"
                    ],
                    label="Detected/Override Document Type",
                    value="Generic Document",
                    interactive=True,
                    info="Auto-detected type shown. You can override if incorrect."
                )
                
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=6,
                    interactive=False
                )
                
                gr.Markdown("### 📊 Document Information")
                doc_info_output = gr.Textbox(
                    label="",
                    lines=8,
                    interactive=False
                )
            
            # Right column - Chat Interface
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Ask Questions")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your documents in any language...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                
                enable_translation = gr.Checkbox(
                    label="Enable multilingual translation",
                    value=True
                )
                
                clear_btn = gr.Button("🗑️ Clear Conversation")
                
                gr.Markdown("### 📚 Sources & Context")
                sources_output = gr.Textbox(
                    label="Retrieved Sources",
                    lines=6,
                    interactive=False
                )
        
        # Bottom section - Insights and Questions
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💡 Suggested Questions")
                questions_output = gr.Textbox(
                    label="",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("## 🔍 Auto-Generated Insights")
                insights_output = gr.Textbox(
                    label="",
                    lines=10,
                    interactive=False
                )
        
        # Footer
        gr.Markdown("""
        ---
        ### 🎯 Features
        - 🌐 **Multilingual**: Support for 100+ languages including Hindi, Tamil, Telugu, Bengali, and more
        - 🔍 **Smart OCR**: Automatic text extraction from scanned documents and images
        - 🧠 **Context-Aware**: Adapts responses based on document type (Resume, Research Paper, Legal, etc.)
        - 📊 **Auto-Insights**: Generates summaries, key points, and suggested questions
        - 🔒 **Privacy-Aware**: Detects and masks PII
        - 💬 **Conversational**: Maintains context across multiple questions
        
        **Powered by**: LangChain + Hugging Face APIs
        """)
        
        # Event handlers
        def process_files(files):
            return app.process_uploaded_files(files)
        
        def answer(query, history, enable_trans):
            return app.answer_question(query, history, enable_trans)
        
        def clear():
            return app.clear_conversation()
        
        def update_doc_type(doc_type):
            return app.update_document_type(doc_type)
        
        # Connect events
        process_btn.click(
            fn=process_files,
            inputs=[file_upload],
            outputs=[status_output, doc_info_output, questions_output, insights_output, doc_type_dropdown]
        )
        
        doc_type_dropdown.change(
            fn=update_doc_type,
            inputs=[doc_type_dropdown],
            outputs=[status_output]
        )
        
        send_btn.click(
            fn=answer,
            inputs=[query_input, chatbot, enable_translation],
            outputs=[chatbot, sources_output]
        ).then(
            lambda: "",
            outputs=[query_input]
        )
        
        query_input.submit(
            fn=answer,
            inputs=[query_input, chatbot, enable_translation],
            outputs=[chatbot, sources_output]
        ).then(
            lambda: "",
            outputs=[query_input]
        )
        
        clear_btn.click(
            fn=clear,
            outputs=[chatbot, sources_output]
        )
    
    return demo


def main():
    """Main entry point"""
    try:
        # Initialize application
        app = OmniDocAI()
        
        # Create and launch UI
        demo = create_ui(app)
        
        logger.info("Launching OmniDoc AI...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    main()
