"""AI-powered enrichment service for generating summaries and extracting complex metadata"""

from typing import Dict, List, Optional, Any
from openai import AzureOpenAI
from config.config import settings
import structlog
import json

logger = structlog.get_logger()

class AIEnrichmentService:
    """Service for AI-powered content enrichment"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        self.deployment_name = settings.azure_openai_deployment_name
        
    async def generate_chunk_summary(self, chunk_content: str, chunk_type: str, 
                                   section_title: str = "") -> str:
        """Generate a concise summary for a chunk"""
        try:
            prompt = f"""Generate a concise summary (max 150 words) of this regulatory text chunk.
Focus on key obligations, rights, penalties, requirements, or definitions mentioned.

Chunk Type: {chunk_type}
Section: {section_title}

Text:
{chunk_content}

Summary:"""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a legal expert summarizing regulatory text. Be precise and focus on actionable content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000  # Increased to allow full summaries
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            # Fallback returns full content as summary
            return chunk_content
    
    async def extract_obligations_and_rights(self, chunk_content: str) -> Dict[str, List[str]]:
        """Extract detailed obligations and rights from chunk"""
        try:
            prompt = f"""Extract obligations and rights from this regulatory text.
Return as JSON with two arrays: "obligations" and "rights".
Be specific and actionable.

Text:
{chunk_content[:2000]}

Format:
{{
    "obligations": ["obligation 1", "obligation 2"],
    "rights": ["right 1", "right 2"]
}}

Result:"""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a legal expert extracting specific obligations and rights. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            # Clean up the response to ensure valid JSON
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            return json.loads(result)
            
        except Exception as e:
            logger.error("Failed to extract obligations/rights", error=str(e))
            return {"obligations": [], "rights": []}
    
    async def extract_penalty_details(self, chunk_content: str) -> List[Dict[str, Any]]:
        """Extract detailed penalty information"""
        try:
            prompt = f"""Extract all penalties and fines mentioned in this text.
Return as JSON array with objects containing: amount, currency, violation, and context.

Text:
{chunk_content[:2000]}

Format:
[
    {{
        "amount": 20000000,
        "currency": "EUR",
        "violation": "description of violation",
        "context": "who can impose and when"
    }}
]

Result:"""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a legal expert extracting penalty information. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            return json.loads(result)
            
        except Exception as e:
            logger.error("Failed to extract penalty details", error=str(e))
            return []
    
    async def extract_definitions(self, chunk_content: str) -> Dict[str, str]:
        """Extract key term definitions"""
        try:
            prompt = f"""Extract formal definitions of terms from this regulatory text.
Return as JSON object with term as key and definition as value.
Only include explicitly defined terms.

Text:
{chunk_content[:2000]}

Format:
{{
    "term1": "definition of term1",
    "term2": "definition of term2"
}}

Result:"""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a legal expert extracting formal definitions. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            return json.loads(result)
            
        except Exception as e:
            logger.error("Failed to extract definitions", error=str(e))
            return {}
    
    async def enrich_chunk_metadata(self, chunk: Dict) -> Dict:
        """Enrich a chunk with AI-generated metadata"""
        content = chunk.get('content', '')
        chunk_type = chunk.get('clause_type', 'general')
        section_title = chunk.get('section_title', '')
        
        # Generate summary if not present
        if not chunk.get('summary'):
            chunk['summary'] = await self.generate_chunk_summary(content, chunk_type, section_title)
        
        # Extract obligations and rights if relevant
        if chunk_type in ['obligation', 'right', 'requirement', 'prohibition']:
            oblig_rights = await self.extract_obligations_and_rights(content)
            if not chunk.get('obligations'):
                chunk['obligations'] = oblig_rights.get('obligations', [])
            if not chunk.get('rights'):
                chunk['rights'] = oblig_rights.get('rights', [])
        
        # Extract penalties if mentioned
        if 'penalty' in content.lower() or 'fine' in content.lower() or 'violation' in content.lower():
            if not chunk.get('penalties') or not chunk['penalties']:
                chunk['penalties'] = await self.extract_penalty_details(content)
        
        # Extract definitions if it's a definition clause
        if chunk_type == 'definition' or 'means' in content.lower() or 'defined as' in content.lower():
            if not chunk.get('definitions'):
                chunk['definitions'] = await self.extract_definitions(content)
        
        return chunk

# Global instance
ai_enrichment_service = AIEnrichmentService()