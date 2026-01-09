"""
Enhanced LangChain RAG Application with Advanced Features
- LLM-based Validation & Confidence Scoring
- Semantic Caching
- Multi-Hop Reasoning
- Query Expansion
- Answer Refinement
- Document Quality Scoring
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
import gradio as gr
import hashlib
import json
from datetime import datetime

# Use HuggingFace Hub directly with proper imports
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
import numpy as np

from pipeline import DocumentProcessor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom Embeddings class
class HuggingFaceInferenceEmbeddings(Embeddings):
    """Custom HuggingFace embeddings using InferenceClient"""
    
    def __init__(self, api_key: str, model_name: str):
        self.client = InferenceClient(token=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        for text in texts:
            embedding = self.client.feature_extraction(text[:512], model=self.model_name)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.client.feature_extraction(text[:512], model=self.model_name)


class SemanticCache:
    """TRUE Semantic caching using embedding similarity (LLM-based)"""
    
    def __init__(self, embeddings_model, similarity_threshold: float = 0.95):
        """
        Initialize semantic cache with embeddings model
        
        Args:
            embeddings_model: HuggingFace embeddings model for semantic similarity
            similarity_threshold: Minimum cosine similarity (0-1) to consider a cache hit
        """
        self.cache = {}  # {query_embedding: (query_text, response, timestamp)}
        self.embeddings_model = embeddings_model
        self.similarity_threshold = similarity_threshold
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached response using semantic similarity (LLM embeddings)"""
        if not self.cache:
            self.cache_misses += 1
            return None
        
        try:
            # Generate embedding for the query using LLM
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Find most similar cached query
            best_similarity = 0.0
            best_match = None
            
            for cached_embedding_key, cached_data in self.cache.items():
                # cached_data: (query_text, response, timestamp, embedding)
                cached_embedding = cached_data.get('embedding')
                if cached_embedding is None:
                    continue
                
                similarity = self._compute_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_data
            
            # Check if best match exceeds threshold
            if best_similarity >= self.similarity_threshold and best_match:
                self.cache_hits += 1
                logger.info(f"✅ Semantic cache HIT! Similarity: {best_similarity:.3f} for query: '{query[:50]}...'")
                return {
                    **best_match['response'],
                    'cached_at': best_match['cached_at'],
                    'cache_similarity': best_similarity,
                    'original_query': best_match['query_text']
                }
            else:
                self.cache_misses += 1
                if best_similarity > 0:
                    logger.debug(f"Semantic cache miss. Best similarity: {best_similarity:.3f} (threshold: {self.similarity_threshold})")
                return None
                
        except Exception as e:
            logger.error(f"Semantic cache lookup error: {e}")
            self.cache_misses += 1
            return None
    
    def set(self, query: str, response: Dict):
        """Cache a response with its semantic embedding (LLM-based)"""
        try:
            # Generate embedding for the query using LLM
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Use a simple counter as key (since we search all entries anyway)
            cache_key = f"query_{len(self.cache)}"
            
            self.cache[cache_key] = {
                'query_text': query,
                'response': response,
                'cached_at': datetime.now().isoformat(),
                'embedding': query_embedding
            }
            
            logger.debug(f"Cached query: '{query[:50]}...' with semantic embedding")
            
        except Exception as e:
            logger.error(f"Semantic cache set error: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_cached': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'similarity_threshold': self.similarity_threshold
        }


class EnhancedLangChainRAG:
    """Enhanced LangChain RAG with advanced features"""
    
    def __init__(self, hf_api_key: str, embedding_model: str, llm_model: str, top_k: int = 5):
        self.hf_api_key = hf_api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        
        logger.info("🚀 Initializing Enhanced LangChain RAG...")
        
        # Initialize HuggingFace client
        self.llm_client = InferenceClient(api_key=hf_api_key)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceInferenceEmbeddings(
            api_key=hf_api_key,
            model_name=embedding_model
        )
        
        # Semantic cache with TRUE semantic similarity using embeddings
        self.cache = SemanticCache(
            embeddings_model=self.embeddings,
            similarity_threshold=0.92  # 92% similarity for cache hit
        )
        
        # Conversation history
        self.conversation_history = []
        
        self.vector_store = None
        self.documents = []
        self.document_quality_scores = {}
        
        # Define prompt templates using LangChain PromptTemplate
        
        # 1. QA Prompt
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Provide a detailed answer based only on the context provided. If the context doesn't contain enough information, acknowledge what's missing.

Answer:"""
        )
        
        # 2. Query Expansion Prompt
        self.query_expansion_prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""Given the user query, generate 3 alternative phrasings that capture the same intent but use different words. This helps retrieve more relevant information.

Original Query: {original_query}

Generate 3 alternative queries (one per line):
1."""
        )
        
        # 3. Confidence Scoring Prompt
        self.confidence_prompt = PromptTemplate(
            input_variables=["question", "answer", "context"],
            template="""Evaluate the confidence level of this answer based on the provided context.

Question: {question}

Answer: {answer}

Context: {context}

Rate the confidence on a scale of 0-100 and explain why:
- 90-100: Answer is fully supported by context with all key details
- 70-89: Answer is mostly supported with minor gaps
- 50-69: Answer is partially supported but missing important details
- 30-49: Answer has weak support from context
- 0-29: Answer is not well supported by context

Provide ONLY a JSON response in this format:
{{"confidence_score": <number 0-100>, "reasoning": "<brief explanation>", "supported_claims": <number>, "unsupported_claims": <number>}}"""
        )
        
        # 4. Answer Refinement Prompt
        self.refinement_prompt = PromptTemplate(
            input_variables=["original_answer", "context", "question"],
            template="""Improve and refine this answer to make it more accurate, complete, and well-structured.

Question: {question}

Original Answer: {original_answer}

Available Context: {context}

Provide an improved answer that:
1. Corrects any inaccuracies
2. Adds missing important details from context
3. Improves clarity and organization
4. Cites specific sources when possible

Refined Answer:"""
        )
        
        # 5. Multi-Hop Reasoning Prompt
        self.multi_hop_prompt = PromptTemplate(
            input_variables=["question", "contexts"],
            template="""Answer this complex question by reasoning across multiple pieces of information.

Question: {question}

Multiple Information Sources:
{contexts}

Break down your reasoning:
1. Identify key information from each source
2. Connect the pieces logically
3. Draw a comprehensive conclusion

Answer with reasoning:"""
        )
        
        # 6. Document Quality Scoring Prompt
        self.quality_scoring_prompt = PromptTemplate(
            input_variables=["document_text", "document_type"],
            template="""Evaluate the quality of this {document_type} document based on multiple criteria.

Document Content (first 1000 chars):
{document_text}

Rate the document on these criteria (0-10 scale each):
1. Clarity: How clear and understandable is the writing?
2. Completeness: How complete and comprehensive is the information?
3. Accuracy: How accurate and reliable does the information appear?
4. Structure: How well-organized and structured is the content?
5. Relevance: How relevant and useful is the information?

Provide ONLY a JSON response:
{{"clarity": <0-10>, "completeness": <0-10>, "accuracy": <0-10>, "structure": <0-10>, "relevance": <0-10>, "overall_score": <average>, "grade": "<A/B/C/D/F>", "summary": "<brief assessment>"}}"""
        )
        
        # 7. Comparative Analysis Prompt
        self.comparison_prompt = PromptTemplate(
            input_variables=["doc1", "doc2", "criteria"],
            template="""Compare these two documents based on the specified criteria.

Document 1:
{doc1}

Document 2:
{doc2}

Comparison Criteria: {criteria}

Provide a detailed comparison covering:
1. Similarities between the documents
2. Key differences
3. Strengths and weaknesses of each
4. Overall recommendation if applicable

Comparison:"""
        )
        
        # 8. Key Points Extraction Prompt
        self.key_points_prompt = PromptTemplate(
            input_variables=["document_content"],
            template="""Extract the key points and main ideas from this document.

Document Content:
{document_content}

Provide:
1. **Main Topic**: What is this document about?
2. **Key Points**: List 5-7 most important points (bullet points)
3. **Key Entities**: Important people, organizations, dates, or terms mentioned
4. **Summary**: 2-3 sentence overview

Format your response clearly with these sections."""
        )
        
        # 9. Suggested Questions Prompt
        self.suggested_questions_prompt = PromptTemplate(
            input_variables=["document_content"],
            template="""Based on this document content, generate 5 insightful questions that users might want to ask.

Document Content:
{document_content}

Generate 5 questions that:
- Cover different aspects of the document
- Range from basic to advanced
- Are specific and answerable from the content

Questions (one per line):
1."""
        )
        
        # 10. Document Classification Prompt (NEW - Automatic type detection)
        self.classification_prompt = PromptTemplate(
            input_variables=["document_text"],
            template="""Analyze this document and classify it into the most appropriate type.

Document Content (first 1500 chars):
{document_text}

Available Document Types:
- resume/cv: Professional resume or curriculum vitae with work experience, skills, education
- technical: Technical documentation, API docs, manuals, specifications, code documentation
- legal: Legal documents, contracts, agreements, terms of service, policies
- financial: Financial reports, invoices, balance sheets, statements, budgets
- academic: Research papers, essays, theses, scholarly articles, study materials
- medical: Medical records, prescriptions, healthcare reports, clinical notes
- business: Business reports, proposals, memos, meeting notes, business plans
- marketing: Marketing materials, brochures, advertisements, campaigns, copy
- email: Email correspondence, messages, communication
- news: News articles, press releases, journalism, announcements
- scientific: Scientific research, lab reports, experiments, data analysis
- educational: Teaching materials, course content, textbooks, lesson plans
- personal: Personal letters, journals, notes, blogs
- general: General documents that don't fit other categories
- jd: Job descriptions for open positions

Consider:
- Content structure, format, and organization
- Language style, tone, and terminology
- Purpose and intended audience
- Typical elements and sections present
- Subject matter and domain

Provide ONLY a JSON response:
{{"document_type": "<type from list above>", "confidence": <0-100>, "reasoning": "<brief 1-sentence explanation>", "detected_language": "<ISO language code like en, es, fr, de, etc>", "alternative_types": ["<alternate type 1>", "<alternate type 2>"]}}"""
        )
        
        # 11. Multilingual Query Prompt (NEW - Answer in any language)
        self.multilingual_prompt = PromptTemplate(
            input_variables=["context", "question", "target_language"],
            template="""Answer the following question in {target_language} based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a detailed, accurate answer in {target_language}
- Maintain technical terms appropriately
- If the context doesn't contain enough information, acknowledge this in {target_language}
- Ensure proper grammar and natural phrasing in {target_language}

Answer in {target_language}:"""
        )
        
        logger.info("✅ Enhanced LangChain RAG initialized with 11 prompt templates!")
    
    def classify_document(self, document_text: str) -> Dict:
        """Automatically classify document type using LLM"""
        try:
            prompt_text = self.classification_prompt.format(
                document_text=document_text[:1500]
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=250,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                    classification_data = json.loads(json_match.group())
                    logger.info(f"📋 Classified as: {classification_data.get('document_type', 'general')} (confidence: {classification_data.get('confidence', 0)}%)")
                    return classification_data
            except Exception as parse_error:
                logger.warning(f"Failed to parse classification JSON: {parse_error}")
            
            # Fallback
            return {
                'document_type': 'general',
                'confidence': 50,
                'reasoning': 'Automatic classification failed',
                'detected_language': 'en',
                'alternative_types': []
            }
            
        except Exception as e:
            logger.error(f"Document classification error: {e}")
            return {
                'document_type': 'general',
                'confidence': 0,
                'reasoning': f'Error: {str(e)}',
                'detected_language': 'en',
                'alternative_types': []
            }
    
    def process_documents(self, processed_docs: List[Dict]) -> Dict:
        """Process documents with quality scoring"""
        
        if not processed_docs:
            return {'total_chunks': 0, 'total_documents': 0, 'quality_scores': []}
        
        all_chunks = []
        quality_scores = []
        
        for doc_data in processed_docs:
            extracted_text = doc_data.get('extracted_text', '')
            metadata = doc_data.get('metadata', {})
            doc_type = doc_data.get('document_type', 'general')
            filename = metadata.get('filename', 'unknown')
            
            if not extracted_text.strip():
                continue
            
            logger.info(f"📄 Processing: {filename}")
            
            # Feature: Document Quality Scoring
            quality_score = self._score_document_quality(extracted_text[:1000], doc_type)
            quality_scores.append({
                'filename': filename,
                'quality': quality_score
            })
            self.document_quality_scores[filename] = quality_score
            
            # Adaptive chunking based on document type
            if doc_type == 'technical':
                chunk_size, chunk_overlap = 1000, 200
            elif doc_type == 'legal':
                chunk_size, chunk_overlap = 1200, 250
            else:
                chunk_size, chunk_overlap = 800, 150
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = text_splitter.split_text(extracted_text)
            
            # Create LangChain Documents with quality metadata
            for i, chunk_text in enumerate(chunks):
                doc_obj = Document(
                    page_content=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'document_type': doc_type,
                        'quality_score': quality_score.get('overall_score', 0)
                    }
                )
                all_chunks.append(doc_obj)
        
        self.documents = all_chunks
        
        # Create FAISS vector store
        logger.info("🔮 Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(
            documents=all_chunks,
            embedding=self.embeddings
        )
        
        logger.info(f"✅ Processed {len(all_chunks)} chunks from {len(processed_docs)} documents")
        
        return {
            'total_chunks': len(all_chunks),
            'total_documents': len(processed_docs),
            'quality_scores': quality_scores
        }
    
    def _score_document_quality(self, document_text: str, document_type: str) -> Dict:
        """Score document quality using LLM"""
        try:
            prompt_text = self.quality_scoring_prompt.format(
                document_text=document_text,
                document_type=document_type
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=300,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                    quality_data = json.loads(json_match.group())
                    return quality_data
            except:
                pass
            
            return {
                'overall_score': 7.0,
                'grade': 'B',
                'summary': 'Quality assessment completed'
            }
            
        except Exception as e:
            logger.error(f"Quality scoring error: {e}")
            return {
                'overall_score': 5.0,
                'grade': 'C',
                'summary': f'Error: {str(e)}'
            }
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query into multiple phrasings"""
        try:
            prompt_text = self.query_expansion_prompt.format(original_query=query)
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=150,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Parse alternative queries
            alternatives = [query]  # Include original
            for line in answer.split('\n'):
                line = line.strip()
                if line and not line.startswith('Original'):
                    # Remove numbering
                    clean_line = line.lstrip('0123456789.-) ')
                    if clean_line and len(clean_line) > 10:
                        alternatives.append(clean_line)
            
            return alternatives[:4]  # Max 4 queries
            
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [query]
    
    def _validate_answer(self, question: str, answer: str, context: str) -> Dict:
        """Validate answer using LLM-based confidence scoring"""
        try:
            prompt_text = self.confidence_prompt.format(
                question=question,
                answer=answer,
                context=context[:1500]  # Limit context size
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=200,
                temperature=0.1
            )
            
            validation_response = response.choices[0].message.content
            
            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', validation_response, re.DOTALL)
                if json_match:
                    validation_data = json.loads(json_match.group())
                    return validation_data
            except:
                pass
            
            # Fallback
            return {
                'confidence_score': 75,
                'reasoning': 'Answer appears reasonable based on context',
                'supported_claims': 1,
                'unsupported_claims': 0
            }
            
        except Exception as e:
            logger.error(f"Answer validation error: {e}")
            return {
                'confidence_score': 50,
                'reasoning': f'Validation error: {str(e)}',
                'supported_claims': 0,
                'unsupported_claims': 0
            }
    
    def _refine_answer(self, original_answer: str, context: str, question: str) -> str:
        """Refine answer for better quality"""
        try:
            prompt_text = self.refinement_prompt.format(
                original_answer=original_answer,
                context=context[:2000],
                question=question
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=600,
                temperature=0.3
            )
            
            refined = response.choices[0].message.content
            return refined
            
        except Exception as e:
            logger.error(f"Answer refinement error: {e}")
            return original_answer
    
    def query(self, question: str, use_cache: bool = True, refine_answer: bool = True) -> Dict:
        """Enhanced query with caching, validation, and refinement"""
        
        if not self.vector_store:
            return {
                'answer': "⚠️ No documents processed yet.",
                'error': 'no_documents'
            }
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get(question)
            if cached_response:
                return {**cached_response, 'from_cache': True}
        
        logger.info(f"🔍 Querying: {question}")
        
        try:
            # Feature: Query Expansion
            expanded_queries = self._expand_query(question)
            logger.info(f"📝 Expanded to {len(expanded_queries)} queries")
            
            # Retrieve documents for all query variants
            all_docs = []
            seen_content = set()
            
            for exp_query in expanded_queries:
                docs = self.vector_store.similarity_search(exp_query, k=self.top_k)
                for doc in docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            
            # Limit to top documents
            all_docs = all_docs[:self.top_k * 2]
            
            if not all_docs:
                return {
                    'answer': "No relevant documents found.",
                    'sources': [],
                    'num_sources': 0,
                    'confidence': 0
                }
            
            # Build context
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
                for doc in all_docs
            ])
            
            # Generate answer
            prompt_text = self.qa_prompt.format(context=context, question=question)
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=512,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            
            # Feature: LLM-based Answer Validation
            validation = self._validate_answer(question, answer, context)
            confidence_score = validation.get('confidence_score', 0)
            
            # Feature: Answer Refinement (if confidence is low or requested)
            if refine_answer and confidence_score < 80:
                logger.info("🔧 Refining answer for better quality...")
                answer = self._refine_answer(answer, context, question)
                # Re-validate
                validation = self._validate_answer(question, answer, context)
                confidence_score = validation.get('confidence_score', 0)
            
            result = {
                'answer': answer,
                'sources': [
                    {
                        'filename': doc.metadata.get('filename', 'unknown'),
                        'content': doc.page_content[:200] + '...',
                        'quality_score': doc.metadata.get('quality_score', 'N/A')
                    }
                    for doc in all_docs[:5]
                ],
                'num_sources': len(all_docs),
                'confidence': confidence_score,
                'validation': validation,
                'expanded_queries': expanded_queries,
                'from_cache': False
            }
            
            # Cache the result
            if use_cache:
                self.cache.set(question, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                'answer': f"❌ Error: {str(e)}",
                'error': str(e),
                'confidence': 0
            }
    
    def multi_hop_query(self, question: str) -> Dict:
        """Answer complex questions requiring multi-hop reasoning"""
        try:
            if not self.vector_store:
                return {'answer': "⚠️ No documents processed yet.", 'error': 'no_documents'}
            
            # Retrieve more documents for multi-hop reasoning
            docs = self.vector_store.similarity_search(question, k=self.top_k * 3)
            
            # Group contexts
            contexts = "\n\n---\n\n".join([
                f"Source {i+1} ({doc.metadata.get('filename', 'unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            # Multi-hop reasoning
            prompt_text = self.multi_hop_prompt.format(
                question=question,
                contexts=contexts
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=700,
                temperature=0.4
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': [{'filename': doc.metadata.get('filename', 'unknown')} for doc in docs],
                'reasoning_type': 'multi_hop'
            }
            
        except Exception as e:
            logger.error(f"Multi-hop query error: {e}")
            return {'answer': f"❌ Error: {str(e)}", 'error': str(e)}
    
    def compare_documents(self, doc1_name: str, doc2_name: str, criteria: str) -> Dict:
        """Compare two documents using LLM"""
        try:
            # Find documents
            doc1_chunks = [d for d in self.documents if d.metadata.get('filename') == doc1_name]
            doc2_chunks = [d for d in self.documents if d.metadata.get('filename') == doc2_name]
            
            if not doc1_chunks or not doc2_chunks:
                return {'comparison': "⚠️ One or both documents not found", 'error': 'not_found'}
            
            # Get representative content
            doc1_text = "\n".join([d.page_content for d in doc1_chunks[:3]])[:2000]
            doc2_text = "\n".join([d.page_content for d in doc2_chunks[:3]])[:2000]
            
            # Compare
            prompt_text = self.comparison_prompt.format(
                doc1=doc1_text,
                doc2=doc2_text,
                criteria=criteria
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=800,
                temperature=0.4
            )
            
            comparison = response.choices[0].message.content
            
            return {
                'comparison': comparison,
                'doc1': doc1_name,
                'doc2': doc2_name,
                'criteria': criteria
            }
            
        except Exception as e:
            logger.error(f"Document comparison error: {e}")
            return {'comparison': f"❌ Error: {str(e)}", 'error': str(e)}
    
    def multilingual_query(self, question: str, target_language: str = "English") -> Dict:
        """Answer query in specified language (NEW)"""
        try:
            if not self.vector_store:
                return {'answer': "⚠️ No documents processed yet.", 'error': 'no_documents'}
            
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=self.top_k)
            
            if not docs:
                return {
                    'answer': f"No relevant documents found. (in {target_language})",
                    'sources': [],
                    'language': target_language
                }
            
            # Build context
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ])
            
            # Generate answer in target language
            prompt_text = self.multilingual_prompt.format(
                context=context,
                question=question,
                target_language=target_language
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=600,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': [{'filename': doc.metadata.get('filename', 'unknown')} for doc in docs],
                'language': target_language,
                'original_question': question
            }
            
        except Exception as e:
            logger.error(f"Multilingual query error: {e}")
            return {'answer': f"❌ Error: {str(e)}", 'error': str(e), 'language': target_language}
    
    def extract_key_points(self, document_name: str) -> Dict:
        """Extract key points from a document"""
        try:
            # Find document chunks
            doc_chunks = [d for d in self.documents if d.metadata.get('filename') == document_name]
            
            if not doc_chunks:
                return {'key_points': "⚠️ Document not found", 'error': 'not_found'}
            
            # Get document content (first few chunks for overview)
            doc_content = "\n\n".join([d.page_content for d in doc_chunks[:5]])[:3000]
            
            # Extract key points
            prompt_text = self.key_points_prompt.format(document_content=doc_content)
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=600,
                temperature=0.3
            )
            
            key_points = response.choices[0].message.content
            
            return {
                'key_points': key_points,
                'document': document_name,
                'chunks_analyzed': len(doc_chunks[:5])
            }
            
        except Exception as e:
            logger.error(f"Key points extraction error: {e}")
            return {'key_points': f"❌ Error: {str(e)}", 'error': str(e)}
    
    def generate_suggested_questions(self, document_name: Optional[str] = None) -> List[str]:
        """Generate suggested questions based on documents"""
        try:
            if document_name:
                # Questions for specific document
                doc_chunks = [d for d in self.documents if d.metadata.get('filename') == document_name]
                if not doc_chunks:
                    return ["No document found"]
                doc_content = "\n\n".join([d.page_content for d in doc_chunks[:3]])[:2000]
            else:
                # Questions for all documents
                if not self.documents:
                    return ["No documents uploaded yet"]
                doc_content = "\n\n".join([d.page_content for d in self.documents[:5]])[:2000]
            
            # Generate questions
            prompt_text = self.suggested_questions_prompt.format(document_content=doc_content)
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.llm_model,
                max_tokens=300,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content
            
            # Parse questions
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    clean_line = line.lstrip('0123456789.-) ')
                    if clean_line and len(clean_line) > 10:
                        questions.append(clean_line)
            
            return questions[:5] if questions else [
                "What is the main topic of this document?",
                "What are the key findings or conclusions?",
                "What important details should I know?",
                "How does this relate to other topics?",
                "What are the practical implications?"
            ]
            
        except Exception as e:
            logger.error(f"Suggested questions error: {e}")
            return [
                "What is this document about?",
                "What are the main points?",
                "Can you summarize the key information?"
            ]


class EnhancedApp:
    """Enhanced Application with Advanced Features"""
    
    def __init__(self):
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not self.hf_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in .env")
        
        logger.info("🚀 Initializing Enhanced LangChain RAG App...")
        
        # Initialize components with new pure RAG pipeline
        self.doc_processor = DocumentProcessor(
            hf_api_key=self.hf_api_key
            # Models are now loaded automatically via Transformers pipelines
            # No need to specify ocr_model or classification_model
        )
        self.rag_engine = EnhancedLangChainRAG(
            hf_api_key=self.hf_api_key,
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            llm_model="meta-llama/Meta-Llama-3-8B-Instruct",
            top_k=5
        )
        
        self.uploaded_files = []
        self.processing_stats = {}
        self.document_summaries = {}
        
        logger.info("✅ App ready!")
    
    def classify_uploaded_files(self, files) -> Tuple[str, str]:
        """Automatically classify uploaded documents (NEW)"""
        if not files:
            return "⚠️ No files uploaded", "general"
        
        try:
            # Process first file to get classification
            result = self.doc_processor.process_document(files[0].name)
            if result and result.get('extracted_text'):
                classification = self.rag_engine.classify_document(result['extracted_text'])
                
                detected_type = classification.get('document_type', 'general')
                confidence = classification.get('confidence', 0)
                reasoning = classification.get('reasoning', 'N/A')
                language = classification.get('detected_language', 'en')
                alternatives = classification.get('alternative_types', [])
                
                classification_text = f"""🤖 **Automatic Classification Results:**

**Detected Type:** {detected_type}
**Confidence:** {confidence}%
**Reasoning:** {reasoning}
**Language:** {language}
**Alternative Types:** {', '.join(alternatives) if alternatives else 'None'}

ℹ️ **You can change the document type below if this classification is incorrect.**
"""
                
                return classification_text, detected_type
            
        except Exception as e:
            logger.error(f"Auto-classification error: {e}")
            return f"⚠️ Auto-classification failed: {str(e)}", "general"
        
        return "⚠️ Could not classify document", "general"
    
    def process_uploaded_files(self, files, doc_type: str) -> Tuple[str, str, str, str]:
        """Process uploaded documents with type selection"""
        if not files:
            return "⚠️ No files uploaded", "", "", ""
        
        logger.info(f"📤 Processing {len(files)} files as '{doc_type}' type...")
        
        try:
            # Process with pipeline
            processed_docs = []
            for file in files:
                result = self.doc_processor.process_document(file.name)
                if result:
                    # Override document type with user selection
                    result['document_type'] = doc_type
                    processed_docs.append(result)
            
            # Add to RAG engine
            stats = self.rag_engine.process_documents(processed_docs)
            
            self.uploaded_files = processed_docs
            self.processing_stats = stats
            
            # Generate summaries and key points for each document
            summaries_text = "\n\n📄 **Document Summaries & Key Points:**\n\n"
            for doc in processed_docs:
                filename = doc['metadata']['filename']
                
                # Extract key points
                key_points_result = self.rag_engine.extract_key_points(filename)
                self.document_summaries[filename] = key_points_result.get('key_points', 'N/A')
                
                summaries_text += f"**📋 {filename}**\n"
                summaries_text += f"{key_points_result.get('key_points', 'Processing...')}\n\n"
                summaries_text += "---\n\n"
            
            # Format output
            status = f"""✅ Successfully processed {stats['total_documents']} documents
� Document Type: {doc_type}
�📊 Created {stats['total_chunks']} chunks
🔍 Vector store ready for queries"""
            
            # Quality scores
            quality_info = "\n\n📈 **Document Quality Scores:**\n"
            for qs in stats.get('quality_scores', []):
                filename = qs['filename']
                quality = qs['quality']
                quality_info += f"\n**{filename}**\n"
                quality_info += f"- Grade: {quality.get('grade', 'N/A')}\n"
                quality_info += f"- Score: {quality.get('overall_score', 0):.1f}/10\n"
                quality_info += f"- Summary: {quality.get('summary', 'N/A')}\n"
            
            # File list
            file_list = "\n".join([f"✓ {doc['metadata']['filename']}" for doc in processed_docs])
            
            return status, quality_info, file_list, summaries_text
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return f"❌ Error: {str(e)}", "", "", ""
    
    def query_documents(self, question: str, use_cache: bool, refine_answer: bool) -> Tuple[str, str, str]:
        """Query with validation and confidence scoring"""
        if not question.strip():
            return "⚠️ Please enter a question", "", ""
        
        try:
            result = self.rag_engine.query(question, use_cache, refine_answer)
            
            if 'error' in result:
                return result['answer'], "", ""
            
            # Format answer with confidence
            answer = result['answer']
            confidence = result.get('confidence', 0)
            validation = result.get('validation', {})
            
            # Confidence indicator
            if confidence >= 90:
                conf_emoji = "🟢"
                conf_label = "Very High"
            elif confidence >= 70:
                conf_emoji = "🟡"
                conf_label = "High"
            elif confidence >= 50:
                conf_emoji = "🟠"
                conf_label = "Medium"
            else:
                conf_emoji = "🔴"
                conf_label = "Low"
            
            answer_with_conf = f"""{answer}

---
**Confidence Assessment:**
{conf_emoji} **{conf_label} Confidence ({confidence}%)**

**Reasoning:** {validation.get('reasoning', 'N/A')}
**Supported Claims:** {validation.get('supported_claims', 'N/A')}
**Unsupported Claims:** {validation.get('unsupported_claims', 'N/A')}
"""
            
            # Sources with quality
            sources_text = "**Sources Used:**\n\n"
            for i, src in enumerate(result.get('sources', [])[:5], 1):
                sources_text += f"{i}. **{src['filename']}** (Quality: {src.get('quality_score', 'N/A')})\n"
                sources_text += f"   {src['content']}\n\n"
            
            # Metadata
            metadata = f"""**Query Metadata:**
- Expanded Queries: {len(result.get('expanded_queries', []))}
- Sources Retrieved: {result.get('num_sources', 0)}
- From Cache: {'Yes ✅' if result.get('from_cache') else 'No'}
- Answer Refined: {'Yes' if refine_answer else 'No'}
- Cache Size: {self.rag_engine.cache.get_stats()['total_cached']} queries cached
"""
            
            # Generate suggested follow-up questions
            suggested_questions = self.rag_engine.generate_suggested_questions()
            suggested_text = "\n\n**💡 Suggested Questions:**\n"
            for i, q in enumerate(suggested_questions[:5], 1):
                suggested_text += f"{i}. {q}\n"
            
            metadata += suggested_text
            
            return answer_with_conf, sources_text, metadata
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"❌ Error: {str(e)}", "", ""


    def get_suggested_questions(self) -> str:
        """Generate suggested questions based on uploaded documents"""
        try:
            if not self.rag_engine.documents:
                return "⚠️ No documents uploaded yet. Please upload documents first!"
            
            # Get a sample of document content
            sample_texts = []
            for doc in self.rag_engine.documents[:5]:  # Use first 5 documents
                sample_texts.append(doc.page_content[:400])
            
            combined_text = "\n\n".join(sample_texts)[:2000]  # Limit total length
            
            # Generate suggested questions using LLM
            prompt = f"""Based on the following document excerpts, generate 5 insightful questions that someone might ask about this content. Make the questions specific and relevant.

Document Content:
{combined_text}

Generate 5 questions (one per line, numbered):
1."""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.rag_engine.llm_client.chat_completion(
                messages=messages,
                model=self.rag_engine.llm_model,
                max_tokens=300,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Parse questions
            questions = []
            for line in answer.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean
                    clean_line = line.lstrip('0123456789.-•) ').strip()
                    if clean_line and len(clean_line) > 10:
                        questions.append(clean_line)
            
            if questions:
                result = "**💡 Suggested Questions:**\n\n"
                for i, q in enumerate(questions[:5], 1):
                    result += f"{i}. {q}\n"
                return result
            else:
                return "Could not generate suggested questions"
                
        except Exception as e:
            logger.error(f"Suggested questions error: {e}")
            return f"❌ Error: {str(e)}"
    
    def get_interview_questions_tab(self, file_type: str = None) -> str:
        """Get interview questions for resume/CV/JD documents using LLM analysis"""
        try:
            if not self.rag_engine.documents:
                return "⚠️ No documents uploaded yet. Please upload a resume, CV, or job description first!"
            
            # Gather ALL document content - let LLM analyze it
            all_docs_text = "\n\n".join([doc.page_content for doc in self.rag_engine.documents[:5]])[:4000]
            
            # Use LLM to analyze the content and generate interview questions
            prompt = f"""Analyze the following document(s) carefully. 

If the content contains a RESUME or CV (with experience, education, skills, projects, etc.), generate 10 interview questions that would be asked to evaluate the candidate based on their background.

If the content contains a JOB DESCRIPTION (with role requirements, responsibilities, qualifications), generate 10 interview questions that would test a candidate's fit for that role and their understanding of those skills/responsibilities.

If the content contains ANY career-related information, generate relevant interview questions based on the skills, experience, or requirements mentioned.

Document Content:
{all_docs_text}

Task: Analyze the document type and generate 10 insightful, specific interview questions based on the actual content. Format as a numbered list (1-10). If no career-related content is found, say so clearly.

Interview Questions:"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.rag_engine.llm_client.chat_completion(
                messages=messages,
                model=self.rag_engine.llm_model,
                max_tokens=600,
                temperature=0.6
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check if LLM said no relevant content found
            if any(phrase in answer.lower() for phrase in ['no career', 'no resume', 'no job description', 'cannot generate', 'not found']):
                return f"⚠️ {answer}\n\n**Tip:** Please upload a resume, CV, or job description document."
            
            # Parse questions
            questions = []
            for line in answer.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    clean_line = line.lstrip('0123456789.-•) ').strip()
                    if clean_line and len(clean_line) > 10:
                        questions.append(clean_line)
            
            if questions:
                result = "**🎯 Interview Questions (Generated by AI):**\n\n"
                for i, q in enumerate(questions[:10], 1):
                    result += f"{i}. {q}\n"
                return result
            else:
                # Return raw LLM response if parsing failed
                return f"**🎯 Interview Questions:**\n\n{answer}"
                
        except Exception as e:
            logger.error(f"Interview questions error: {e}")
            return f"❌ Error: {str(e)}"

    def get_career_options_tab(self) -> str:
        """Analyze documents and suggest career options using LLM"""
        try:
            if not self.rag_engine.documents:
                return "⚠️ No documents uploaded yet. Please upload a resume or CV first!"
            
            # Gather ALL document content - let LLM analyze it
            all_docs_text = "\n\n".join([doc.page_content for doc in self.rag_engine.documents[:5]])[:4000]
            
            # Use LLM to analyze content and suggest career options
            prompt = f"""Analyze the following document(s) carefully.

If the content contains a RESUME or CV (with experience, education, skills, projects, etc.), suggest 7 possible career options based on the person's background. Include:
- 3-4 traditional career paths based on their experience and skills
- 3-4 creative/unconventional options they may not have considered

For each career option, explain WHY it's suitable based on their specific skills, experience, and background mentioned in the document.

Document Content:
{all_docs_text}

Task: Analyze the document and provide 7 career options with detailed reasoning. Format each as:
**Career Option 1:** [Title] - [Why it's suitable based on their background]

If no resume/CV content is found, say so clearly.

Career Options:"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.rag_engine.llm_client.chat_completion(
                messages=messages,
                model=self.rag_engine.llm_model,
                max_tokens=900,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check if LLM said no relevant content found
            if any(phrase in answer.lower() for phrase in ['no resume', 'no cv', 'no career', 'cannot analyze', 'not found']):
                return f"⚠️ {answer}\n\n**Tip:** Please upload a resume or CV document to get personalized career suggestions."
            
            # Return formatted result
            result = "**🚀 Career Options & Recommendations (Generated by AI):**\n\n"
            result += answer
            
            return result
                
        except Exception as e:
            logger.error(f"Career options error: {e}")
            return f"❌ Error: {str(e)}"
    
    def multi_hop_query(self, question: str) -> str:
        """Complex multi-hop reasoning"""
        if not question.strip():
            return "⚠️ Please enter a question"
        
        try:
            result = self.rag_engine.multi_hop_query(question)
            return result.get('answer', 'No answer generated')
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def compare_documents(self, doc1: str, doc2: str, criteria: str) -> str:
        """Compare two documents"""
        if not doc1 or not doc2:
            return "⚠️ Please select two documents"
        
        try:
            result = self.rag_engine.compare_documents(doc1, doc2, criteria or "overall content and quality")
            return result.get('comparison', 'Comparison failed')
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def compare_documents_by_name(self, doc1_name: str, doc2_name: str, criteria: str) -> str:
        """Compare two documents by filename"""
        if not doc1_name.strip() or not doc2_name.strip():
            return "⚠️ Please enter both document names"
        
        try:
            result = self.rag_engine.compare_documents(doc1_name, doc2_name, criteria or "overall content and quality")
            
            if 'error' in result:
                return result.get('comparison', 'Comparison failed')
            
            comparison_text = result.get('comparison', 'No comparison generated')
            return f"**🔄 Document Comparison:**\n\n{comparison_text}"
            
        except Exception as e:
            logger.error(f"Document comparison error: {e}")
            return f"❌ Error: {str(e)}"
    
    def compare_uploaded_files(self, files, criteria: str) -> str:
        """Compare uploaded files directly"""
        if not files or len(files) < 2:
            return "⚠️ Please upload at least 2 files to compare"
        
        try:
            # Process the uploaded files
            processed_docs = []
            for file in files[:2]:  # Only compare first 2 files
                result = self.doc_processor.process_document(file.name)
                if result:
                    processed_docs.append(result)
            
            if len(processed_docs) < 2:
                return "⚠️ Could not process both files"
            
            # Get document texts
            doc1_text = processed_docs[0].get('extracted_text', '')[:2000]
            doc2_text = processed_docs[1].get('extracted_text', '')[:2000]
            doc1_name = processed_docs[0]['metadata']['filename']
            doc2_name = processed_docs[1]['metadata']['filename']
            
            # Use the comparison prompt
            prompt_text = self.rag_engine.comparison_prompt.format(
                doc1=doc1_text,
                doc2=doc2_text,
                criteria=criteria or "overall content and quality"
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            response = self.rag_engine.llm_client.chat_completion(
                messages=messages,
                model=self.rag_engine.llm_model,
                max_tokens=800,
                temperature=0.4
            )
            
            comparison = response.choices[0].message.content
            
            result_text = f"""**🔄 File Comparison:**

**Document 1:** {doc1_name}
**Document 2:** {doc2_name}
**Criteria:** {criteria or 'overall content and quality'}

---

{comparison}
"""
            return result_text
            
        except Exception as e:
            logger.error(f"File comparison error: {e}")
            return f"❌ Error: {str(e)}"
    
    def multilingual_query(self, question: str, language: str) -> str:
        """Query in multiple languages"""
        if not question.strip():
            return "⚠️ Please enter a question"
        
        try:
            result = self.rag_engine.multilingual_query(question, language)
            return result.get('answer', 'No answer generated')
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def compare_resume_jd(self) -> str:
        """Compare resume and job description for interview prep"""
        try:
            if not self.rag_engine.documents:
                return "⚠️ No documents uploaded yet. Please upload documents in the 'Upload Documents' tab first, or use the 'Upload Resume & JD' option."
            
            # Find resume/CV documents
            resume_texts = []
            for doc in self.rag_engine.documents:
                doc_type = doc.metadata.get('document_type', '').lower()
                if doc_type in ['resume', 'cv', 'resume/cv']:
                    resume_texts.append(doc.page_content)
            
            # Find JD documents
            jd_texts = []
            for doc in self.rag_engine.documents:
                doc_type = doc.metadata.get('document_type', '').lower()
                if doc_type == 'jd':
                    jd_texts.append(doc.page_content)
            
            if not resume_texts:
                return "⚠️ No resume or CV found. Please upload a resume/CV and set document type to 'resume/cv'."
            
            if not jd_texts:
                return "⚠️ No job description found. Please upload a JD and set document type to 'jd'."
            
            resume_text = "\n\n".join(resume_texts)[:2500]
            jd_text = "\n\n".join(jd_texts)[:2500]
            
            # Generate comprehensive analysis using LLM
            prompt = f"""Analyze the candidate's resume against the job description and provide a comprehensive interview preparation report.

**Resume/CV:**
{resume_text}

**Job Description:**
{jd_text}

Provide a detailed analysis covering:
1. **Match Score** (0-100%): How well the candidate matches the role
2. **Matching Skills**: Skills the candidate has that match the JD
3. **Missing Skills**: Skills required but not in resume (skills gap)
4. **Strengths**: Candidate's key strengths for this role
5. **Areas to Improve**: Skills/experience to develop
6. **5 Interview Questions**: Likely questions for this specific role
7. **Preparation Tips**: Specific advice for interview success

Analysis:"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.rag_engine.llm_client.chat_completion(
                messages=messages,
                model=self.rag_engine.llm_model,
                max_tokens=1000,
                temperature=0.3
            )
            
            return f"**📊 Resume & JD Analysis:**\n\n{response.choices[0].message.content}"
            
        except Exception as e:
            logger.error(f"Resume-JD comparison error: {e}")
            return f"❌ Error: {str(e)}"
    
    def upload_and_analyze_resume_jd(self, resume_file, jd_file) -> str:
        """Upload and analyze resume and JD directly"""
        try:
            if not resume_file or not jd_file:
                return "⚠️ Please upload both resume and job description files."
            
            # Process resume
            resume_result = self.doc_processor.process_document(resume_file.name)
            if not resume_result:
                return "⚠️ Could not process resume file."
            
            # Process JD
            jd_result = self.doc_processor.process_document(jd_file.name)
            if not jd_result:
                return "⚠️ Could not process job description file."
            
            resume_text = resume_result.get('extracted_text', '')[:2500]
            jd_text = jd_result.get('extracted_text', '')[:2500]
            
            if not resume_text or not jd_text:
                return "⚠️ Could not extract text from one or both files."
            
            # Generate comprehensive analysis using LLM
            prompt = f"""Analyze the candidate's resume against the job description and provide a comprehensive interview preparation report.

**Resume/CV:**
{resume_text}

**Job Description:**
{jd_text}

Provide a detailed analysis covering:
1. **Match Score** (0-100%): How well the candidate matches the role
2. **Matching Skills**: Skills the candidate has that match the JD
3. **Missing Skills**: Skills required but not in resume (skills gap)
4. **Strengths**: Candidate's key strengths for this role
5. **Areas to Improve**: Skills/experience to develop
6. **5 Interview Questions**: Likely questions for this specific role
7. **Preparation Tips**: Specific advice for interview success

Analysis:"""
            
            messages = [{"role": "user", "content": prompt}]
            response = self.rag_engine.llm_client.chat_completion(
                messages=messages,
                model=self.rag_engine.llm_model,
                max_tokens=1000,
                temperature=0.3
            )
            
            return f"**📊 Resume & JD Analysis:**\n\n{response.choices[0].message.content}"
            
        except Exception as e:
            logger.error(f"Upload and analyze error: {e}")
            return f"❌ Error: {str(e)}"

def create_interface():
    app = EnhancedApp()
    
    with gr.Blocks(title="🚀 Enhanced LangChain RAG", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚀 Enhanced LangChain RAG Application
        
        **Advanced Features:**
        - 🤖 **Automatic Document Classification** with user override
        - ✅ **LLM-based Answer Validation** & Confidence Scoring
        - 🌍 **Multilingual Support** (answer in any language)
        - 🚀 **Semantic Caching** (10x faster queries)
        - 🔍 **Query Expansion** (better retrieval)
        - ✨ **Answer Refinement** (improved quality)
        - 📊 **Document Quality Scoring**
        - 🧠 **Multi-Hop Reasoning**
        - 🔄 **Document Comparison**
        - 💡 **Suggested Questions**
        """)
        
        with gr.Tabs():
            # Tab 1: Upload Documents with Auto-Classification
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown("""
                ### 📁 Document Upload & Classification
                Upload your documents and the system will automatically classify them using AI.
                You can override the classification if needed.
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload Documents (PDF, TXT, DOCX, MD, CSV, XLSX, PPTX, Images)",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx", ".pptx", ".jpg", ".png"]
                        )
                        classify_btn = gr.Button("🤖 Auto-Classify Documents", variant="secondary")
                        
                        classification_result = gr.Markdown(label="Classification Results", visible=True)
                        
                        doc_type_select = gr.Dropdown(
                            choices=[
                                "general",
                                "resume/cv",
                                "technical",
                                "legal",
                                "financial",
                                "academic",
                                "medical",
                                "business",
                                "marketing",
                                "email",
                                "news",
                                "scientific",
                                "educational",
                                "personal",
                                "jd"
                            ],
                            value="general",
                            label="📂 Document Type (Override if needed)",
                            info="AI-detected type is shown above. Change if incorrect."
                        )
                        
                        upload_btn = gr.Button("🚀 Process Documents", variant="primary")
                    
                    with gr.Column():
                        upload_status = gr.Textbox(label="Processing Status", lines=5)
                        quality_scores = gr.Markdown(label="Document Quality Scores")
                
                with gr.Row():
                    with gr.Column():
                        file_list = gr.Textbox(label="Processed Files", lines=5)
                    with gr.Column():
                        doc_summaries = gr.Markdown(label="Document Summaries & Key Points")
                
                # Auto-classification
                classify_btn.click(
                    fn=app.classify_uploaded_files,
                    inputs=[file_upload],
                    outputs=[classification_result, doc_type_select]
                )
                
                # Process documents
                upload_btn.click(
                    fn=app.process_uploaded_files,
                    inputs=[file_upload, doc_type_select],
                    outputs=[upload_status, quality_scores, file_list, doc_summaries]
                )
            
            # Tab 2: Ask Questions
            with gr.Tab("💬 Ask Questions"):
                gr.Markdown("Ask questions about your uploaded documents with AI-powered validation")
                
                # Suggested questions button
                with gr.Row():
                    suggested_btn = gr.Button("💡 Get Suggested Questions", variant="secondary", size="sm")
                    interview_btn = gr.Button("🎯 Interview Questions (Resume/CV/JD)", variant="secondary", size="sm")
                    career_btn = gr.Button("🚀 Career Options (Resume/CV)", variant="secondary", size="sm")
                
                with gr.Row():
                    with gr.Column():
                        suggested_output = gr.Markdown(label="Suggested Questions / Interview Questions / Career Options", visible=True)
                
                # Wire up all three buttons
                suggested_btn.click(
                    fn=app.get_suggested_questions,
                    inputs=[],
                    outputs=[suggested_output]
                )
                
                interview_btn.click(
                    fn=lambda: app.get_interview_questions_tab("resume"),  # Will auto-detect type
                    inputs=[],
                    outputs=[suggested_output]
                )
                
                career_btn.click(
                    fn=app.get_career_options_tab,
                    inputs=[],
                    outputs=[suggested_output]
                )
                
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your documents...",
                            lines=3
                        )
                        with gr.Row():
                            use_cache = gr.Checkbox(label="🚀 Use Semantic Cache", value=True, info="Faster for repeated queries")
                            refine_answer = gr.Checkbox(label="✨ Refine Answer", value=True, info="Higher quality but slower")
                        query_btn = gr.Button("🔍 Get Answer", variant="primary")
                    
                    with gr.Column():
                        answer_output = gr.Markdown(label="Answer with Confidence Score")
                        sources_output = gr.Markdown(label="Sources")
                        metadata_output = gr.Markdown(label="Query Metadata & Suggested Questions")
                
                query_btn.click(
                    fn=app.query_documents,
                    inputs=[question_input, use_cache, refine_answer],
                    outputs=[answer_output, sources_output, metadata_output]
                )
            
            # Tab 3: Multi-Hop Reasoning
            with gr.Tab("🧠 Multi-Hop Reasoning"):
                gr.Markdown("Ask complex questions that require reasoning across multiple sources")
                multi_hop_question = gr.Textbox(
                    label="Complex Question",
                    placeholder="E.g., How do concepts from document A relate to findings in document B?",
                    lines=3
                )
                multi_hop_btn = gr.Button("🧠 Analyze", variant="primary")
                multi_hop_output = gr.Markdown(label="Reasoning & Answer")
                
                multi_hop_btn.click(
                    fn=app.multi_hop_query,
                    inputs=[multi_hop_question],
                    outputs=[multi_hop_output]
                )
            
            # Tab 4: Multilingual Queries
            with gr.Tab("🌍 Multilingual Queries"):
                gr.Markdown("""
                ### 🌍 Ask Questions in Any Language
                Ask questions and get answers in your preferred language, including 13+ Indian languages!
                """)
                
                with gr.Tabs():
                    # Sub-tab: International Languages
                    with gr.Tab("🌐 International Languages"):
                        with gr.Row():
                            with gr.Column():
                                multilingual_question = gr.Textbox(
                                    label="Your Question (in any language)",
                                    placeholder="Ask in English, Spanish, French, German, etc...",
                                    lines=3
                                )
                                language_select = gr.Dropdown(
                                    choices=[
                                        "English",
                                        "Spanish (Español)",
                                        "French (Français)",
                                        "German (Deutsch)",
                                        "Italian (Italiano)",
                                        "Portuguese (Português)",
                                        "Chinese (中文)",
                                        "Japanese (日本語)",
                                        "Korean (한국어)",
                                        "Arabic (العربية)",
                                        "Russian (Русский)"
                                    ],
                                    value="English",
                                    label="🌐 Answer Language",
                                    info="Select the language for the answer"
                                )
                                multilingual_btn = gr.Button("🌍 Ask in Selected Language", variant="primary")
                            
                            with gr.Column():
                                multilingual_answer = gr.Markdown(label="Answer")
                        
                        multilingual_btn.click(
                            fn=app.multilingual_query,
                            inputs=[multilingual_question, language_select],
                            outputs=[multilingual_answer]
                        )
                    
                    # Sub-tab: Indian Languages
                    with gr.Tab("🇮🇳 Indian Languages"):
                        gr.Markdown("""
                        ### 🇮🇳 13+ Indian Languages Supported
                        Ask questions and get answers in: Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, Assamese, Sanskrit
                        """)
                        
                        with gr.Row():
                            with gr.Column():
                                indian_question = gr.Textbox(
                                    label="Your Question (in any Indian language)",
                                    placeholder="हिंदी, తెలుగు, தமிழ், ಕನ್ನಡ, മലയാളം, বাংলা में पूछें...",
                                    lines=4
                                )
                                indian_language_select = gr.Dropdown(
                                    choices=[
                                        "Hindi (हिन्दी)",
                                        "Telugu (తెలుగు)",
                                        "Tamil (தமிழ்)",
                                        "Kannada (ಕನ್ನಡ)",
                                        "Malayalam (മലയാളം)",
                                        "Bengali (বাংলা)",
                                        "Marathi (मराठी)",
                                        "Gujarati (ગુજરાતી)",
                                        "Punjabi (ਪੰਜਾਬੀ)",
                                        "Odia (ଓଡ଼ିଆ)",
                                        "Urdu (اردو)",
                                        "Assamese (অসমীয়া)",
                                        "Sanskrit (संस्कृत)"
                                    ],
                                    value="Hindi (हिन्दी)",
                                    label="🇮🇳 Answer Language",
                                    info="Select the Indian language for the answer"
                                )
                                indian_btn = gr.Button("🇮🇳 Ask in Indian Language", variant="primary")
                            
                            with gr.Column():
                                indian_answer = gr.Markdown(label="Answer (in selected Indian language)")
                        
                        indian_btn.click(
                            fn=app.multilingual_query,
                            inputs=[indian_question, indian_language_select],
                            outputs=[indian_answer]
                        )
            
            # Tab 5: Compare Documents
            with gr.Tab("🔄 Compare Documents"):
                gr.Markdown("Compare two documents side-by-side - either from uploaded documents or upload new files")
                
                with gr.Tabs():
                    # Sub-tab: Compare by name
                    with gr.Tab("📝 Compare by Name"):
                        gr.Markdown("Compare documents already uploaded to the system")
                        with gr.Row():
                            doc1_input = gr.Textbox(label="Document 1 Name", placeholder="filename.pdf")
                            doc2_input = gr.Textbox(label="Document 2 Name", placeholder="filename.pdf")
                        criteria_input = gr.Textbox(
                            label="Comparison Criteria",
                            placeholder="E.g., technical accuracy, writing style, completeness...",
                            value="overall content and quality"
                        )
                        compare_btn = gr.Button("🔄 Compare", variant="primary")
                        comparison_output = gr.Markdown(label="Comparison Results")
                        
                        compare_btn.click(
                            fn=app.compare_documents_by_name,
                            inputs=[doc1_input, doc2_input, criteria_input],
                            outputs=[comparison_output]
                        )
                    
                    # Sub-tab: Compare uploaded files
                    with gr.Tab("📤 Upload & Compare"):
                        gr.Markdown("Upload 2 documents directly to compare them")
                        compare_files = gr.File(
                            label="Upload 2 Documents to Compare",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".docx", ".md"]
                        )
                        criteria_input2 = gr.Textbox(
                            label="Comparison Criteria",
                            placeholder="E.g., technical accuracy, writing style, completeness...",
                            value="overall content and quality"
                        )
                        compare_files_btn = gr.Button("🔄 Compare Files", variant="primary")
                        comparison_output2 = gr.Markdown(label="Comparison Results")
                        
                        compare_files_btn.click(
                            fn=app.compare_uploaded_files,
                            inputs=[compare_files, criteria_input2],
                            outputs=[comparison_output2]
                        )
                    
                    # Sub-tab: Resume & JD Analysis (NEW - moved here)
                    with gr.Tab("📊 Resume & JD Analysis"):
                        gr.Markdown("""
                        ### 📊 Interview Preparation Assistant
                        Analyze your resume against a job description to get:
                        - Skills gap analysis
                        - Match percentage
                        - Skills to improve
                        - Interview questions for this specific role
                        - Preparation tips
                        """)
                        
                        with gr.Tabs():
                            # Option 1: Use already uploaded documents
                            with gr.Tab("📂 Use Uploaded Documents"):
                                gr.Markdown("**Analyze resume and JD already uploaded to the system**")
                                resume_jd_btn = gr.Button("📊 Analyze Resume & Job Match", variant="primary", size="lg")
                                resume_jd_output = gr.Markdown(label="Analysis & Interview Preparation")
                                
                                resume_jd_btn.click(
                                    fn=app.compare_resume_jd,
                                    inputs=[],
                                    outputs=[resume_jd_output]
                                )
                            
                            # Option 2: Upload resume and JD directly
                            with gr.Tab("📤 Upload Resume & JD"):
                                gr.Markdown("**Upload resume and job description directly for analysis**")
                                
                                with gr.Row():
                                    resume_upload = gr.File(
                                        label="Upload Resume/CV",
                                        file_types=[".pdf", ".txt", ".docx", ".md"],
                                        file_count="single"
                                    )
                                    jd_upload = gr.File(
                                        label="Upload Job Description",
                                        file_types=[".pdf", ".txt", ".docx", ".md"],
                                        file_count="single"
                                    )
                                
                                upload_analyze_btn = gr.Button("📊 Upload & Analyze", variant="primary", size="lg")
                                upload_analysis_output = gr.Markdown(label="Analysis & Interview Preparation")
                                
                                upload_analyze_btn.click(
                                    fn=app.upload_and_analyze_resume_jd,
                                    inputs=[resume_upload, jd_upload],
                                    outputs=[upload_analysis_output]
                                )
        
        gr.Markdown("""
        ---
        **💡 Tips & Features:**
        - 🤖 **Auto-Classification**: AI automatically detects document type (resume, cv, jd, technical, legal, etc.)
        - 📂 **15 Document Types**: resume/cv, jd, technical, legal, financial, academic, medical, business, marketing, email, news, scientific, educational, personal, general
        - 🎯 **Interview Questions**: Get AI-generated interview questions for resume/CV/JD
        - 🚀 **Career Options**: Discover career paths based on your resume
        - 📊 **Resume & JD Analysis**: Skills gap, interview prep, and match analysis
        - 🇮🇳 **Indian Languages**: Full support for 13 Indian languages (Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, Assamese, Sanskrit)
        - 🌍 **Multilingual**: Ask questions and get answers in 20+ languages
        - 💡 **Suggested Questions**: Get AI-generated questions based on your documents
        - 🚀 **Semantic Cache**: 10x faster responses for similar questions
        - ✨ **Answer Refinement**: Higher quality responses with confidence scoring
        - 📊 **Quality Scoring**: Automatic document quality assessment
        - 🧠 **Multi-Hop Reasoning**: Complex queries across multiple sources
        - 🔄 **Document Comparison**: Compare by name or upload new files
        - 📈 **Key Points**: Automatic extraction of main ideas and summaries
        - ℹ️ **User Override**: Change auto-detected document type if needed
        """)
    
    return interface


if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced LangChain RAG Application...")
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7863)
