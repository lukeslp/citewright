"""
LLM Analyzer for CiteWright.

Provides optional LLM-based metadata extraction for difficult cases
where traditional API lookups fail.

Supports multiple providers:
- Ollama (local, free)
- OpenAI
- Anthropic
- Google Gemini
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .api.clients import PaperMetadata
from .utils.config import get_config, AI_PROVIDERS

logger = logging.getLogger(__name__)


# System prompt for metadata extraction
EXTRACTION_PROMPT = """You are an expert academic librarian. Your task is to extract bibliographic metadata from the provided text, which is the beginning of an academic paper or book.

Extract the following information if available:
- title: The full title of the paper or book
- authors: List of author names (full names if available)
- year: Publication year (4-digit number)
- abstract: Brief summary if visible
- venue: Journal name, conference name, or publisher
- keywords: Subject keywords or categories

If you cannot determine a field with confidence, leave it as null.

Respond ONLY with a valid JSON object in this exact format:
{
  "title": "string or null",
  "authors": ["list of strings"] or [],
  "year": number or null,
  "abstract": "string or null",
  "venue": "string or null",
  "keywords": ["list of strings"] or []
}

Do not include any explanation or text outside the JSON object."""


class AIAnalyzer:
    """
    Uses LLMs to extract metadata from document text.
    
    This is the Tier 4 fallback strategy, used when identifier-based
    and search-based methods fail.
    """
    
    def __init__(self):
        self.config = get_config()
        self._client = None
    
    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is not None:
            return self._client
        
        ai_config = self.config.get_ai_config(vision=False)
        provider = ai_config["provider"]
        
        if provider == "ollama":
            self._client = self._create_ollama_client(ai_config)
        else:
            self._client = self._create_openai_compatible_client(ai_config)
        
        return self._client
    
    def _create_ollama_client(self, config: Dict) -> Any:
        """Create an Ollama client."""
        try:
            from openai import OpenAI
            
            return OpenAI(
                base_url=config["base_url"],
                api_key="ollama",  # Ollama doesn't require a real key
            )
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            return None
    
    def _create_openai_compatible_client(self, config: Dict) -> Any:
        """Create an OpenAI-compatible client for various providers."""
        try:
            from openai import OpenAI
            
            api_key = config["api_key"]
            if not api_key:
                logger.error(f"No API key found for provider: {config['provider']}")
                return None
            
            # Handle provider-specific base URLs
            provider = config["provider"]
            base_url = config["base_url"]
            
            # Anthropic requires a different client
            if provider == "anthropic":
                try:
                    from anthropic import Anthropic
                    return Anthropic(api_key=api_key)
                except ImportError:
                    logger.error("anthropic package not installed. Run: pip install anthropic")
                    return None
            
            # Gemini uses a different API structure
            if provider == "gemini":
                # Use OpenAI-compatible endpoint if available
                return OpenAI(
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key=api_key,
                )
            
            return OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            return None
    
    def analyze_text(self, text: str, filename: str = "") -> Optional[PaperMetadata]:
        """
        Analyze text using an LLM to extract metadata.
        
        Args:
            text: Document text to analyze.
            filename: Original filename (for context).
            
        Returns:
            PaperMetadata or None if analysis fails.
        """
        if not self.config.ai_enabled:
            logger.debug("LLM analysis is disabled")
            return None
        
        client = self._get_client()
        if client is None:
            logger.error("Failed to create LLM client")
            return None
        
        ai_config = self.config.get_ai_config()
        provider = ai_config["provider"]
        model = ai_config["model"]
        
        # Prepare the prompt
        user_message = f"Filename: {filename}\n\nDocument text:\n{text[:4000]}"
        
        try:
            if provider == "anthropic":
                return self._analyze_with_anthropic(client, model, user_message)
            else:
                return self._analyze_with_openai(client, model, user_message)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
    
    def _analyze_with_openai(self, client, model: str, user_message: str) -> Optional[PaperMetadata]:
        """Analyze using OpenAI-compatible API."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content
        return self._parse_response(content)
    
    def _analyze_with_anthropic(self, client, model: str, user_message: str) -> Optional[PaperMetadata]:
        """Analyze using Anthropic API."""
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system=EXTRACTION_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        
        content = response.content[0].text
        return self._parse_response(content)
    
    def _parse_response(self, content: str) -> Optional[PaperMetadata]:
        """Parse the LLM response into PaperMetadata."""
        try:
            # Try to extract JSON from the response
            content = content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            data = json.loads(content)
            
            # Validate required field
            if not data.get("title"):
                logger.warning("LLM response missing title")
                return None
            
            return PaperMetadata(
                title=data["title"],
                authors=data.get("authors", []),
                year=data.get("year"),
                abstract=data.get("abstract"),
                venue=data.get("venue"),
                keywords=data.get("keywords", []),
                source="ai_analysis",
                confidence=0.6,  # Lower confidence for LLM-extracted data
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return None
    
    def analyze_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze an image using a vision model.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with extracted information or None.
        """
        if not self.config.ai_enabled:
            logger.debug("LLM analysis is disabled")
            return None
        
        client = self._get_client()
        if client is None:
            return None
        
        ai_config = self.config.get_ai_config(vision=True)
        provider = ai_config["provider"]
        model = ai_config["model"]
        
        # Read and encode image
        import base64
        
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return None
        
        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")
        
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": image_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image briefly. What is the main subject? Suggest a descriptive filename (without extension).",
                                },
                            ],
                        }
                    ],
                )
                content = response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_data}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image briefly. What is the main subject? Suggest a descriptive filename (without extension).",
                                },
                            ],
                        }
                    ],
                    max_tokens=500,
                )
                content = response.choices[0].message.content
            
            return {
                "description": content,
                "source": "ai_vision",
            }
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return None


def is_ai_available() -> bool:
    """Check if LLM analysis is available and configured."""
    config = get_config()
    
    if not config.ai_enabled:
        return False
    
    ai_config = config.get_ai_config()
    provider = ai_config["provider"]
    
    # Ollama is always "available" (local)
    if provider == "ollama":
        return True
    
    # Check if API key is set
    return ai_config["api_key"] is not None
