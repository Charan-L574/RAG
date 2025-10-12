"""
RAG Engine with Multilingual Support
Handles embeddings, vector storage, retrieval, and LLM generation
Uses Hugging Face Inference API via transformers (no local model downloads)
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualRAGEngine:
    """RAG Engine using Hugging Face Inference API (no local model downloads)"""
    
    def __init__(
        self,
        hf_api_key: str,
        embedding_model: str,
        llm_model: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5
    ):
        self.hf_api_key = hf_api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize HuggingFace Inference Client for embeddings (no local download)
        logger.info(f"Initializing HF Inference Client for embeddings: {embedding_model}")
        self.inference_client = InferenceClient(token=hf_api_key)
        
        # Load tokenizer from transformers (lightweight, only downloads config files)
        logger.info(f"Loading tokenizer from transformers: {llm_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using default.")
            self.tokenizer = None
        
        # Store LLM model name for InferenceClient text generation
        self.llm_model = llm_model
        logger.info(f"LLM ready: Will use InferenceClient.text_generation() with transformers tokenizer")
        self.llm_model_id = llm_model
        
        # Vector store
        self.documents = []  # List of Document objects
        self.embeddings_matrix = None
        self.faiss_index = None
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info("RAG Engine initialized successfully with HF Inference API (no local models)!")
    
    def add_documents(self, processed_docs: List[Dict]) -> int:
        """
        Add processed documents to the RAG system
        
        Args:
            processed_docs: List of documents with extracted_text, metadata, pages
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Adding {len(processed_docs)} documents to RAG system...")
        
        all_chunks = []
        
        for doc in processed_docs:
            extracted_text = doc.get('extracted_text', '')
            metadata = doc.get('metadata', {})
            doc_type = doc.get('document_type', 'Generic Document')
            
            if not extracted_text:
                logger.warning(f"Skipping empty document: {metadata.get('filename', 'unknown')}")
                continue
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(extracted_text)
            
            # Create Document objects with metadata
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        'chunk_id': i,
                        'document_type': doc_type,
                        'total_chunks': len(chunks)
                    }
                )
                all_chunks.append(chunk_doc)
        
        self.documents.extend(all_chunks)
        
        # Generate embeddings for new chunks
        self._update_embeddings()
        
        logger.info(f"Added {len(all_chunks)} chunks to vector store")
        return len(all_chunks)
    
    def _update_embeddings(self):
        """Generate embeddings for all documents and update FAISS index"""
        if not self.documents:
            logger.warning("No documents to embed")
            return
        
        logger.info(f"Generating embeddings for {len(self.documents)} chunks...")
        
        # Get embeddings for all documents
        texts = [doc.page_content for doc in self.documents]
        embeddings = self._get_embeddings_batch(texts)
        
        if embeddings is None or len(embeddings) == 0:
            logger.error("Failed to generate embeddings")
            return
        
        # Convert to numpy array
        self.embeddings_matrix = np.array(embeddings).astype('float32')
        
        # Create/update FAISS index
        dimension = self.embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings_matrix)
        
        logger.info(f"FAISS index updated with {len(self.documents)} vectors")
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 5) -> Optional[List[List[float]]]:
        """Get embeddings for texts using HuggingFace InferenceClient (no local models)"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {self.embedding_model}...")
            
            all_embeddings = []
            
            # Process texts individually for better reliability
            for i, text in enumerate(texts):
                try:
                    # Use InferenceClient feature_extraction method (correct API)
                    result = self.inference_client.feature_extraction(
                        text=text,
                        model=self.embedding_model
                    )
                    
                    # The result is typically a numpy array or list
                    # Convert to list if it's a numpy array
                    if hasattr(result, 'tolist'):
                        embedding = result.tolist()
                    elif isinstance(result, list):
                        embedding = result
                    else:
                        # Try converting to list
                        embedding = list(result)
                    
                    # Handle different nesting levels
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # Check if it's already a flat vector of numbers
                        if isinstance(embedding[0], (int, float)):
                            all_embeddings.append(embedding)
                        # If nested once [[emb]]
                        elif isinstance(embedding[0], list) and len(embedding[0]) > 0:
                            if isinstance(embedding[0][0], (int, float)):
                                all_embeddings.append(embedding[0])
                            # If nested twice [[[emb]]]
                            elif isinstance(embedding[0][0], list):
                                all_embeddings.append(embedding[0][0])
                            else:
                                logger.error(f"Unexpected nesting at index {i}")
                                return None
                        else:
                            logger.error(f"Unexpected format at index {i}")
                            return None
                    else:
                        logger.error(f"Empty or invalid embedding at index {i}")
                        return None
                    
                    # Log progress
                    if (i + 1) % 5 == 0:
                        logger.info(f"  ✓ Processed {i + 1}/{len(texts)} embeddings")
                        
                except Exception as e:
                    logger.error(f"Error generating embedding for text {i}: {e}")
                    # Log the first 100 chars of the problematic text for debugging
                    logger.error(f"  Problematic text preview: {text[:100]}...")
                    return None
            
            logger.info(f"✅ Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings if len(all_embeddings) == len(texts) else None
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return None
    
    def _get_single_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text using HuggingFace InferenceClient (no local models)"""
        try:
            # Use InferenceClient feature_extraction method
            result = self.inference_client.feature_extraction(
                text=text,
                model=self.embedding_model
            )
            
            # Convert to list if it's a numpy array
            if hasattr(result, 'tolist'):
                embedding = result.tolist()
            elif isinstance(result, list):
                embedding = result
            else:
                embedding = list(result)
            
            # Handle different response formats
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], (int, float)):
                    # Direct embedding vector
                    return embedding
                elif isinstance(embedding[0], list) and len(embedding[0]) > 0:
                    if isinstance(embedding[0][0], (int, float)):
                        # Wrapped [[embedding]]
                        return embedding[0]
                    elif isinstance(embedding[0][0], list):
                        # Triple nested [[[embedding]]]
                        return embedding[0][0]
            
            logger.error(f"Unexpected embedding format for query: {type(embedding)}")
            return None
                
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None
    
    def retrieve_relevant_chunks(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query with ADVANCED RAG TECHNIQUES:
        - Query expansion (generate related queries)
        - Hybrid search (semantic + keyword matching)
        - Result reranking by relevance
        
        Args:
            query: User query
            k: Number of chunks to retrieve (default: self.top_k)
            
        Returns:
            List of dicts with content, metadata, and score
        """
        if k is None:
            k = self.top_k
        
        if not self.faiss_index or not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        # ADVANCED TECHNIQUE 1: Query Expansion
        # Generate semantically similar queries to improve retrieval
        expanded_queries = self._expand_query(query)
        logger.info(f"🔍 Query expansion: {len(expanded_queries)} queries")
        
        # Collect results from all expanded queries
        all_results = {}  # Use dict to deduplicate by chunk_id
        
        for expanded_query in expanded_queries:
            # Get query embedding
            query_embedding = self._get_single_embedding(expanded_query)
            
            if query_embedding is None:
                logger.warning(f"Failed to generate embedding for: {expanded_query}")
                continue
            
            # Convert to numpy array
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS
            search_k = min(k * 2, len(self.documents))  # Get more candidates for reranking
            distances, indices = self.faiss_index.search(query_vector, search_k)
            
            # Collect results with scores
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    chunk_key = f"{doc.metadata.get('filename', '')}_{doc.metadata.get('chunk_id', idx)}"
                    
                    # Keep best score for each chunk
                    similarity_score = 1 / (1 + dist)  # Convert distance to similarity
                    
                    if chunk_key not in all_results or similarity_score > all_results[chunk_key]['score']:
                        all_results[chunk_key] = {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'score': similarity_score,
                            'distance': float(dist)
                        }
        
        # ADVANCED TECHNIQUE 2: Hybrid Search - Boost keyword matches
        results_list = list(all_results.values())
        query_keywords = set(query.lower().split())
        
        for result in results_list:
            content_words = set(result['content'].lower().split())
            keyword_overlap = len(query_keywords & content_words) / max(len(query_keywords), 1)
            
            # Boost score if keywords match
            result['score'] = result['score'] * (1 + 0.3 * keyword_overlap)
            result['keyword_match'] = keyword_overlap
        
        # ADVANCED TECHNIQUE 3: Rerank by combined score
        results_list.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top k results
        top_results = results_list[:k]
        
        logger.info(f"✅ Retrieved {len(top_results)} chunks (scores: {[f'{r['score']:.3f}' for r in top_results[:3]]})")
        
        return top_results
    
    def _expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        ADVANCED RAG: Expand query with related phrasings to improve retrieval
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries to generate
            
        Returns:
            List of queries including original + expanded versions
        """
        queries = [query]  # Always include original
        
        # Simple rule-based expansion (can be enhanced with LLM in future)
        query_lower = query.lower()
        
        # Technique 1: Add synonyms/variations
        variations = []
        
        if "what" in query_lower or "which" in query_lower:
            variations.append(query.replace("What", "List").replace("what", "list").replace("Which", "Identify").replace("which", "identify"))
        
        if "skills" in query_lower:
            variations.append(query.replace("skills", "competencies and skills"))
            variations.append(query.replace("skills", "technical abilities"))
        
        if "responsibilities" in query_lower:
            variations.append(query.replace("responsibilities", "duties and responsibilities"))
            variations.append(query.replace("responsibilities", "key tasks"))
        
        if "requirements" in query_lower:
            variations.append(query.replace("requirements", "qualifications and requirements"))
        
        # Technique 2: Add context-specific rephrasing
        if "?" in query:
            statement = query.replace("?", "").replace("What are", "The").replace("What is", "The")
            variations.append(statement)
        
        # Add valid variations
        queries.extend([v for v in variations if v != query][:max_expansions - 1])
        
        return queries[:max_expansions]
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        document_type: Optional[str] = None,
        language: str = "en",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using LLM based on retrieved context
        
        Args:
            query: User question
            context_chunks: Retrieved relevant chunks
            document_type: Type of document for context-aware prompting
            language: Target language for response
            custom_prompt: Optional custom prompt template
            
        Returns:
            Generated answer
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Build the prompt
        try:
            logger.info("="*60)
            logger.info("STARTING LLM GENERATION - NO FALLBACK MODE")
            logger.info("="*60)
            
            # Log the context we're working with
            logger.info(f"Context length: {len(context)} characters")
            logger.info(f"Context preview (first 300 chars): {context[:300]}")
            
            # CRITICAL: Use conversational endpoint for better results
            logger.info(f"Calling InferenceClient for LLM: {self.llm_model}")
            logger.info(f"Query: {query}")
            
            try:
                # Prepare the context - keep it detailed but manageable
                # For education queries, we need ALL the details
                context_to_send = context[:3000] if len(context) > 3000 else context
                
                logger.info(f"Sending {len(context_to_send)} chars of context to LLM")
                
                # Use chat_completion with clear instructions
                response = self.inference_client.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a precise document analyst. Follow these rules strictly:
1. READ ALL the context provided carefully
2. Extract COMPLETE information - include ALL details like:
   - Full degree names and majors
   - Institution names
   - Years, dates, CGPA/percentages
   - ALL items in lists
3. ORGANIZE information clearly with bullet points or sections
4. NEVER truncate or summarize education history - include EVERYTHING
5. If the context has multiple educational qualifications, list ALL of them
6. Provide a comprehensive, well-formatted answer"""
                        },
                        {
                            "role": "user",
                            "content": f"""Document Context:
{context_to_send}

Question: {query}

Important: Provide a COMPLETE answer including ALL relevant details from the context. Do not skip any information."""
                        }
                    ],
                    model=self.llm_model,
                    max_tokens=500,
                    temperature=0.3,  # Lower for more accurate, focused responses
                )
                
                logger.info(f"LLM Response received: {type(response)}")
                
                # Extract answer from chat completion
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    answer = response.choices[0].message.content.strip()
                    logger.info(f"✅ SUCCESSFULLY GENERATED ANSWER ({len(answer)} chars)")
                    logger.info(f"Full Answer: {answer}")
                    
                    if len(answer) > 20:
                        return answer
                    else:
                        raise ValueError(f"Answer too short: {len(answer)} characters")
                else:
                    raise ValueError(f"Invalid response structure: {response}")
                    
            except Exception as llm_error:
                logger.error(f"❌ LLM GENERATION FAILED: {type(llm_error).__name__}")
                logger.error(f"Error details: {str(llm_error)}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                
                # NO FALLBACK - Return error message forcing user to check
                return f"""❌ LLM Generation Failed

Error: {str(llm_error)}

The LLM could not generate a proper answer. This indicates:
1. Model might not be available
2. Context might be too complex
3. API might have rate limits

Please check the logs and try:
- Using a different question format
- Uploading a clearer document
- Waiting a moment and trying again

Raw context was retrieved but LLM validation is required."""
            
        except Exception as e:
            logger.error(f"Error in generate_answer: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating answer: {str(e)}"
    
    def _extract_answer_from_context(self, query: str, context: str, document_type: Optional[str] = None) -> str:
        """
        FALLBACK: Extract answer directly from context when LLM fails
        Uses simple text extraction and formatting
        """
        logger.info("Using extraction-based fallback to answer query")
        
        # Extract relevant sentences based on query keywords
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Split context into sentences
        sentences = []
        for line in context.split('\n'):
            line = line.strip()
            if len(line) > 20:  # Meaningful content
                sentences.append(line)
        
        # Score sentences by keyword overlap
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            overlap = sum(1 for word in query_words if word in sent_lower)
            if overlap > 0:
                scored_sentences.append((overlap, sent))
        
        # Sort by relevance
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Build answer from top sentences
        if scored_sentences:
            answer_parts = ["Based on the document:\n"]
            for score, sent in scored_sentences[:5]:  # Top 5 relevant sentences
                if not sent.startswith('---'):  # Skip separators
                    answer_parts.append(f"• {sent}")
            
            return "\n".join(answer_parts)
        else:
            # If no keyword matches, return first meaningful content
            meaningful = [s for s in sentences if not s.startswith('---') and len(s) > 30]
            if meaningful:
                return f"Based on the document:\n\n{meaningful[0]}\n\nFor more specific information, please try rephrasing your question."
            else:
                return "I found relevant content but couldn't extract a specific answer. Please try rephrasing your question."
    
    def _build_context(self, chunks: List[Dict], max_length: int = 3500) -> str:
        """Build context string from retrieved chunks with more detail"""
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Add source information with more detail
            source_info = f"\n--- Document Excerpt {i} ---\n"
            source_info += f"Source: {metadata.get('filename', 'Unknown')}"
            if 'page_number' in metadata:
                source_info += f" (Page {metadata['page_number']})"
            if 'document_type' in metadata:
                source_info += f" | Type: {metadata['document_type']}"
            source_info += f"\n{content}\n"
            
            chunk_text = source_info
            
            if current_length + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "".join(context_parts)
    
    def _build_custom_prompt(
        self,
        query: str,
        context: str,
        document_type: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """Build custom prompt for LLM based on document type and language"""
        
        # Check if query is about skills, roadmap, or analysis for Job Descriptions
        query_lower = query.lower()
        is_skills_query = any(keyword in query_lower for keyword in [
            'skill', 'requirement', 'qualification', 'roadmap', 'prepare', 'crack', 'learn'
        ])
        
        # Base instruction - CRITICAL: Emphasize validation and summarization
        if document_type == "Job Description" and is_skills_query:
            base_instruction = (
                "You are an expert career counselor and technical recruiter. "
                "Analyze the job description thoroughly and provide comprehensive, actionable guidance.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. READ and VALIDATE all information from the context below\n"
                "2. DO NOT just copy raw text - SUMMARIZE and SYNTHESIZE\n"
                "3. Extract ALL required and preferred skills mentioned\n"
                "4. Categorize skills (technical, soft skills, tools, frameworks)\n"
                "5. Provide a detailed learning roadmap with specific steps\n"
                "6. Give realistic time estimates for skill acquisition\n\n"
            )
        else:
            base_instruction = (
                "You are an expert document analyst with deep comprehension skills. "
                "Your task is to VALIDATE and SYNTHESIZE information, NOT just repeat raw text.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. READ the context carefully and UNDERSTAND the information\n"
                "2. VALIDATE that your answer directly addresses the question\n"
                "3. SUMMARIZE in clear, concise language - DO NOT copy-paste chunks\n"
                "4. STRUCTURE your response with proper formatting\n"
                "5. Focus on RELEVANT information only\n"
                "6. If multiple pieces of info exist, SYNTHESIZE them into a coherent answer\n\n"
            )
        
        # Add language instruction
        if language != "en":
            base_instruction += f"Please respond in {language}.\n\n"
        
        # Add document-type specific instructions
        if document_type:
            type_instructions = {
                "Job Description": (
                    "This is a Job Description. Focus on:\n"
                    "- Required skills, experience, and qualifications\n"
                    "- Key responsibilities and expectations\n"
                    "- Company culture and team structure if mentioned\n"
                    "- Career growth opportunities\n"
                    "For skills/roadmap questions, provide actionable learning paths."
                ),
                "Resume or CV": (
                    "This is a Resume/CV. Focus on:\n"
                    "- Candidate's skills, experience, and education\n"
                    "- Key achievements and projects\n"
                    "- Technical proficiencies and certifications\n"
                    "- Career progression and strengths"
                ),
                "Research Paper or Academic Article": (
                    "This is a Research Paper. Focus on:\n"
                    "- Research methodology and approach\n"
                    "- Key findings and conclusions\n"
                    "- Technical concepts explained clearly\n"
                    "- Implications and applications"
                ),
                "Legal Document or Contract": (
                    "This is a Legal Document. Focus on:\n"
                    "- Explaining legal terms in simple language\n"
                    "- Key obligations, rights, and conditions\n"
                    "- Important clauses and their implications\n"
                    "- Potential risks or considerations"
                ),
                "Invoice or Financial Report": (
                    "This is a Financial Document. Focus on:\n"
                    "- Financial figures, totals, and breakdowns\n"
                    "- Payment terms and conditions\n"
                    "- Key financial information and calculations"
                ),
                "Textbook or Educational Material": (
                    "This is Educational Material. Focus on:\n"
                    "- Explaining concepts clearly with examples\n"
                    "- Providing comprehensive understanding\n"
                    "- Breaking down complex topics\n"
                    "- Relating concepts to practical applications"
                ),
            }
            
            if document_type in type_instructions:
                base_instruction += f"{type_instructions[document_type]}\n\n"
        
        # Construct full prompt
        prompt = f"""{base_instruction}Context from Documents:
{context}

Question: {query}

Comprehensive Answer:"""
        
        return prompt
    
    def get_statistics(self) -> Dict:
        """Get statistics about the RAG system"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            'index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }
    
    def clear(self):
        """Clear all documents and reset the system"""
        self.documents = []
        self.embeddings_matrix = None
        self.faiss_index = None
        logger.info("RAG system cleared")
    
    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        if self.faiss_index:
            faiss.write_index(self.faiss_index, filepath)
            # Also save documents metadata
            docs_data = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in self.documents
            ]
            with open(filepath + '.docs.json', 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        if os.path.exists(filepath):
            self.faiss_index = faiss.read_index(filepath)
            # Load documents metadata
            docs_file = filepath + '.docs.json'
            if os.path.exists(docs_file):
                with open(docs_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                self.documents = [
                    Document(page_content=d['content'], metadata=d['metadata'])
                    for d in docs_data
                ]
            logger.info(f"Index loaded from {filepath}")
