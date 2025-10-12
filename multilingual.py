"""
Multilingual Support Module
Handles language detection, translation, and cross-language queries
"""

import logging
from typing import Optional, Dict
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultilingualProcessor:
    """Handles multilingual operations including detection and translation"""
    
    # Language code mappings
    LANGUAGE_NAMES = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'ur': 'Urdu',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'ru': 'Russian',
        'pt': 'Portuguese'
    }
    
    def __init__(
        self,
        hf_api_key: str,
        language_detection_model: str,
        translation_model: str
    ):
        self.hf_api_key = hf_api_key
        self.language_detection_model = language_detection_model
        self.translation_model = translation_model
        self.hf_api_headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    def detect_language(self, text: str) -> Dict[str, str]:
        """
        Detect the language of the input text
        
        Args:
            text: Input text
            
        Returns:
            Dict with 'code' and 'name' of detected language
        """
        if not text or len(text.strip()) < 3:
            return {'code': 'en', 'name': 'English', 'confidence': 0.0}
        
        try:
            # Truncate to first 200 characters for detection
            text_sample = text[:200]
            
            api_url = f"https://api-inference.huggingface.co/models/{self.language_detection_model}"
            payload = {"inputs": text_sample}
            
            response = requests.post(
                api_url,
                headers=self.hf_api_headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse result - usually returns list of predictions
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # List of predictions
                        top_prediction = result[0][0]
                        lang_code = top_prediction.get('label', 'en')
                        confidence = top_prediction.get('score', 0.0)
                    elif isinstance(result[0], dict):
                        lang_code = result[0].get('label', 'en')
                        confidence = result[0].get('score', 0.0)
                    else:
                        lang_code = 'en'
                        confidence = 0.0
                    
                    lang_name = self.LANGUAGE_NAMES.get(lang_code, lang_code.upper())
                    
                    return {
                        'code': lang_code,
                        'name': lang_name,
                        'confidence': confidence
                    }
            else:
                logger.warning(f"Language detection API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
        
        # Default to English
        return {'code': 'en', 'name': 'English', 'confidence': 0.0}
    
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if translation fails
        """
        # If same language, return original
        if source_lang == target_lang:
            return text
        
        # If target is English and source is not, translate to English
        # Otherwise, try to use available translation models
        
        try:
            # For now, using a general multilingual to English model
            # In production, you'd route to specific model based on language pair
            if target_lang == 'en':
                model = self.translation_model
            else:
                # For Indian languages, might need specific models
                model = self._get_translation_model(source_lang, target_lang)
            
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            payload = {"inputs": text}
            
            response = requests.post(
                api_url,
                headers=self.hf_api_headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse translation result
                if isinstance(result, list) and len(result) > 0:
                    if 'translation_text' in result[0]:
                        return result[0]['translation_text']
                    elif isinstance(result[0], str):
                        return result[0]
                elif isinstance(result, dict) and 'translation_text' in result:
                    return result['translation_text']
            else:
                logger.warning(f"Translation API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
        
        return None
    
    def _get_translation_model(self, source_lang: str, target_lang: str) -> str:
        """
        Get appropriate translation model for language pair
        
        This is a simplified version. In production, you'd have a mapping
        of language pairs to specific models.
        """
        # Map of some common translation models
        translation_models = {
            ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
            ('ta', 'en'): 'Helsinki-NLP/opus-mt-ta-en',
            ('te', 'en'): 'Helsinki-NLP/opus-mt-te-en',
            ('bn', 'en'): 'Helsinki-NLP/opus-mt-bn-en',
            ('mr', 'en'): 'Helsinki-NLP/opus-mt-mr-en',
            ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi',
            ('en', 'ta'): 'Helsinki-NLP/opus-mt-en-ta',
            ('en', 'te'): 'Helsinki-NLP/opus-mt-en-te',
            ('en', 'bn'): 'Helsinki-NLP/opus-mt-en-bn',
        }
        
        # Try to find specific model
        model = translation_models.get((source_lang, target_lang))
        
        if model:
            return model
        
        # Fallback to general multilingual model
        return self.translation_model
    
    def process_multilingual_query(
        self,
        query: str,
        document_language: str = 'en'
    ) -> Dict:
        """
        Process a multilingual query
        
        Returns:
            Dict with original query, detected language, and translated query (if needed)
        """
        # Detect query language
        query_lang = self.detect_language(query)
        
        result = {
            'original_query': query,
            'query_language': query_lang,
            'translated_query': None,
            'needs_translation': False
        }
        
        # If query language differs from document language, translate
        if query_lang['code'] != document_language:
            translated = self.translate_text(
                query,
                query_lang['code'],
                document_language
            )
            
            if translated:
                result['translated_query'] = translated
                result['needs_translation'] = True
        
        return result
    
    def translate_response(
        self,
        response: str,
        target_language: str
    ) -> Optional[str]:
        """
        Translate a response to target language
        
        Args:
            response: Response text in English (assumed)
            target_language: Target language code
            
        Returns:
            Translated response or original if translation fails
        """
        if target_language == 'en':
            return response
        
        translated = self.translate_text(response, 'en', target_language)
        return translated if translated else response
    
    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """Get dictionary of supported languages"""
        return MultilingualProcessor.LANGUAGE_NAMES.copy()
    
    @staticmethod
    def get_indian_languages() -> Dict[str, str]:
        """Get dictionary of supported Indian languages"""
        indian_langs = {
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }
        return indian_langs


class ContextAwarePromptBuilder:
    """Builds context-aware prompts based on document type"""
    
    PROMPT_TEMPLATES = {
        "Resume or CV": """You are analyzing a resume/CV. Focus on:
- Professional experience and skills
- Educational qualifications
- Key achievements and strengths
- Career progression

Context:
{context}

Question: {query}

Provide a detailed, professional analysis:""",
        
        "Research Paper or Academic Article": """You are analyzing a research paper. Focus on:
- Research methodology and approach
- Key findings and results
- Conclusions and implications
- Technical accuracy

Context:
{context}

Question: {query}

Provide an academic, well-structured answer:""",
        
        "Legal Document or Contract": """You are analyzing a legal document. Focus on:
- Key terms and conditions
- Legal obligations and rights
- Important clauses and provisions
- Explain in simple, clear language

Context:
{context}

Question: {query}

Provide a clear explanation:""",
        
        "Invoice or Financial Report": """You are analyzing a financial document. Focus on:
- Financial figures and totals
- Transaction details
- Key financial metrics
- Numerical accuracy

Context:
{context}

Question: {query}

Provide a precise, numerical answer:""",
        
        "Textbook or Educational Material": """You are explaining educational content. Focus on:
- Clear concept explanation
- Examples when helpful
- Step-by-step breakdowns
- Educational value

Context:
{context}

Question: {query}

Provide a clear, educational explanation:""",
        
        "Generic Document": """You are analyzing a document. Answer based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    }
    
    @staticmethod
    def get_prompt(document_type: str, context: str, query: str) -> str:
        """Get appropriate prompt template for document type"""
        template = ContextAwarePromptBuilder.PROMPT_TEMPLATES.get(
            document_type,
            ContextAwarePromptBuilder.PROMPT_TEMPLATES["Generic Document"]
        )
        
        return template.format(context=context, query=query)
