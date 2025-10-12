"""
Advanced Features Module
Includes auto-question generation, PII masking, knowledge graphs, and insights
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates relevant questions based on document content"""
    
    def __init__(self, hf_api_key: str, llm_model: str):
        self.hf_api_key = hf_api_key
        self.llm_model = llm_model
        self.hf_api_headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    def generate_questions(
        self,
        document_text: str,
        document_type: str,
        num_questions: int = 5
    ) -> List[str]:
        """
        Generate suggested questions based on document content
        
        Args:
            document_text: Text content of document
            document_type: Type of document
            num_questions: Number of questions to generate
            
        Returns:
            List of suggested questions
        """
        # Truncate document for API call
        text_sample = " ".join(document_text.split()[:500])
        
        # Build prompt based on document type
        prompt = self._build_question_generation_prompt(text_sample, document_type, num_questions)
        
        # Call LLM
        try:
            api_url = f"https://api-inference.huggingface.co/models/{self.llm_model}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.8,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                api_url,
                headers=self.hf_api_headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', '')
                else:
                    generated_text = ''
                
                # Parse questions from generated text
                questions = self._parse_questions(generated_text)
                return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
        
        # Fallback to template-based questions
        return self._get_template_questions(document_type)
    
    def _build_question_generation_prompt(
        self,
        text: str,
        doc_type: str,
        num: int
    ) -> str:
        """Build prompt for question generation"""
        prompt = f"""Based on the following document, generate {num} relevant and insightful questions that a user might want to ask about it.

Document Type: {doc_type}

Document Content:
{text}

Generate {num} specific questions (numbered 1-{num}):
1."""
        return prompt
    
    def _parse_questions(self, generated_text: str) -> List[str]:
        """Parse questions from generated text"""
        questions = []
        
        # Look for numbered questions
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1. Question?" or "1) Question?"
            match = re.match(r'^\d+[\.\)]\s*(.+\?)\s*$', line)
            if match:
                questions.append(match.group(1))
            elif line.endswith('?') and len(line) > 10:
                questions.append(line)
        
        return questions
    
    def _get_template_questions(self, doc_type: str) -> List[str]:
        """Get template questions based on document type"""
        templates = {
            "Resume or CV": [
                "What are the candidate's key skills?",
                "What is the candidate's educational background?",
                "What is the candidate's total work experience?",
                "What are the candidate's major achievements?",
                "What roles has the candidate held?"
            ],
            "Research Paper or Academic Article": [
                "What is the main research question or hypothesis?",
                "What methodology was used in this study?",
                "What are the key findings?",
                "What are the conclusions of this paper?",
                "What are the limitations mentioned?"
            ],
            "Legal Document or Contract": [
                "What are the key terms and conditions?",
                "What are the main obligations of each party?",
                "What is the duration or validity period?",
                "What are the termination conditions?",
                "What are the payment terms?"
            ],
            "Invoice or Financial Report": [
                "What is the total amount?",
                "Who is the vendor/supplier?",
                "What is the payment due date?",
                "What items or services are included?",
                "What are the tax details?"
            ],
            "Textbook or Educational Material": [
                "What are the main topics covered?",
                "Can you explain the key concepts?",
                "What are the important definitions?",
                "Are there any examples provided?",
                "What are the learning objectives?"
            ],
            "Generic Document": [
                "What is this document about?",
                "What are the main points?",
                "Can you summarize this document?",
                "What are the key details?",
                "What information is most important?"
            ]
        }
        
        return templates.get(doc_type, templates["Generic Document"])


class PIIMasker:
    """Detects and masks Personally Identifiable Information"""
    
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    }
    
    @staticmethod
    def mask_pii(text: str, mask_char: str = 'X') -> Tuple[str, List[Dict]]:
        """
        Mask PII in text
        
        Args:
            text: Input text
            mask_char: Character to use for masking
            
        Returns:
            Tuple of (masked_text, list of detected PII items)
        """
        masked_text = text
        detected_pii = []
        
        for pii_type, pattern in PIIMasker.PII_PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                original = match.group()
                # Create mask with same length
                mask = mask_char * len(original)
                masked_text = masked_text.replace(original, mask, 1)
                
                detected_pii.append({
                    'type': pii_type,
                    'original': original,
                    'position': match.start()
                })
        
        return masked_text, detected_pii
    
    @staticmethod
    def detect_pii(text: str) -> List[Dict]:
        """
        Detect PII without masking
        
        Returns:
            List of detected PII items with type and position
        """
        _, detected = PIIMasker.mask_pii(text)
        return detected


class DocumentInsights:
    """Generates automatic insights from documents"""
    
    def __init__(self, hf_api_key: str, llm_model: str):
        self.hf_api_key = hf_api_key
        self.llm_model = llm_model
        self.hf_api_headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    def generate_insights(
        self,
        document_text: str,
        document_type: str,
        metadata: Dict
    ) -> Dict:
        """
        Generate automatic insights from document
        
        Returns:
            Dict with summary, key_points, entities, statistics
        """
        insights = {
            'summary': '',
            'key_points': [],
            'statistics': self._get_statistics(document_text, metadata),
            'entities': []
        }
        
        # Generate summary
        insights['summary'] = self._generate_summary(document_text, document_type)
        
        # Extract key points
        insights['key_points'] = self._extract_key_points(document_text, document_type)
        
        # Extract entities (simple regex-based for now)
        insights['entities'] = self._extract_entities(document_text)
        
        return insights
    
    def _generate_summary(self, text: str, doc_type: str) -> str:
        """Generate document summary"""
        # Truncate for API
        text_sample = " ".join(text.split()[:800])
        
        prompt = f"""Summarize the following {doc_type} in 2-3 sentences:

{text_sample}

Summary:"""
        
        try:
            api_url = f"https://api-inference.huggingface.co/models/{self.llm_model}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.5
                }
            }
            
            response = requests.post(
                api_url,
                headers=self.hf_api_headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                elif isinstance(result, dict):
                    return result.get('generated_text', '').strip()
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
        
        # Fallback: first few sentences
        sentences = text.split('.')[:3]
        return '. '.join(sentences) + '.'
    
    def _extract_key_points(self, text: str, doc_type: str) -> List[str]:
        """Extract key points from document"""
        # Type-specific extraction
        if doc_type == "Resume or CV":
            return self._extract_resume_points(text)
        elif doc_type == "Research Paper or Academic Article":
            return self._extract_research_points(text)
        else:
            # Generic key points - extract sentences with important keywords
            important_keywords = ['important', 'key', 'significant', 'critical', 'essential', 'main', 'primary']
            sentences = text.split('.')
            key_points = []
            
            for sentence in sentences[:20]:  # Check first 20 sentences
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    key_points.append(sentence.strip())
                    if len(key_points) >= 5:
                        break
            
            return key_points
    
    def _extract_resume_points(self, text: str) -> List[str]:
        """Extract key points from resume"""
        points = []
        
        # Look for experience
        if 'experience' in text.lower():
            points.append("Professional experience available")
        
        # Look for education
        if any(edu in text.lower() for edu in ['bachelor', 'master', 'phd', 'degree', 'university']):
            points.append("Educational qualifications documented")
        
        # Look for skills section
        if 'skills' in text.lower() or 'technologies' in text.lower():
            points.append("Technical skills and competencies listed")
        
        return points
    
    def _extract_research_points(self, text: str) -> List[str]:
        """Extract key points from research paper"""
        points = []
        
        sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        for section in sections:
            if section in text.lower():
                points.append(f"{section.capitalize()} section present")
        
        return points
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Simple entity extraction using regex patterns"""
        entities = []
        
        # Organizations (capitalized words followed by Inc, Corp, Ltd, etc.)
        org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|Company)\b'
        orgs = re.findall(org_pattern, text)
        for org in set(orgs):
            entities.append({'text': org, 'type': 'ORGANIZATION'})
        
        # Dates
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text)
        for date in set(dates):
            entities.append({'text': date, 'type': 'DATE'})
        
        # Money
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        money = re.findall(money_pattern, text)
        for amount in set(money):
            entities.append({'text': amount, 'type': 'MONEY'})
        
        return entities[:20]  # Limit to 20 entities
    
    def _get_statistics(self, text: str, metadata: Dict) -> Dict:
        """Get basic statistics about the document"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'pages': metadata.get('total_pages', 1),
            'file_type': metadata.get('file_type', 'unknown')
        }


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
    
    def add_interaction(self, query: str, response: str, context: List[Dict]):
        """Add a query-response pair to history"""
        self.history.append({
            'query': query,
            'response': response,
            'context': context,
            'timestamp': None  # Could add datetime if needed
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        if last_n:
            return self.history[-last_n:]
        return self.history
    
    def get_context_for_query(self, query: str) -> str:
        """Build context string from recent history for follow-up questions"""
        if not self.history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for item in self.history[-3:]:  # Last 3 interactions
            context_parts.append(f"Q: {item['query']}")
            context_parts.append(f"A: {item['response'][:200]}")  # Truncate
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []


class DocumentComparator:
    """Compare multiple documents"""
    
    @staticmethod
    def compare_documents(documents: List[Dict]) -> Dict:
        """
        Compare multiple documents
        
        Args:
            documents: List of processed documents
            
        Returns:
            Comparison insights
        """
        if len(documents) < 2:
            return {'error': 'Need at least 2 documents to compare'}
        
        comparison = {
            'total_documents': len(documents),
            'document_types': {},
            'size_comparison': [],
            'common_themes': []
        }
        
        # Count document types
        for doc in documents:
            doc_type = doc.get('document_type', 'Unknown')
            comparison['document_types'][doc_type] = comparison['document_types'].get(doc_type, 0) + 1
        
        # Compare sizes
        for doc in documents:
            metadata = doc.get('metadata', {})
            comparison['size_comparison'].append({
                'filename': metadata.get('filename', 'Unknown'),
                'word_count': len(doc.get('extracted_text', '').split()),
                'pages': metadata.get('total_pages', 1)
            })
        
        return comparison
