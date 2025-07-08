"""
Prompt Templates - Loads and manages prompts from consolidated JSON file
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    """Types of query intents"""
    COMPARISON = "comparison"
    SPECIFIC_REQUIREMENT = "specific_requirement"
    GENERAL_INQUIRY = "general_inquiry"
    CLARIFICATION = "clarification"
    TIMELINE = "timeline"
    COMPLIANCE_CHECK = "compliance_check"
    DEFINITION = "definition"


@dataclass
class PromptTemplate:
    """Structure for prompt templates"""
    intent: QueryIntent
    system_prompt: str
    user_prompt_template: str
    response_guidelines: List[str]
    example_format: Optional[str] = None
    metadata_instructions: Optional[str] = None


class PromptTemplateManager:
    """Manages prompt templates loaded from consolidated JSON file"""
    
    def __init__(self, prompts_file: Optional[str] = None):
        # Set prompts file path
        if prompts_file:
            self.prompts_file = Path(prompts_file)
        else:
            # Default to prompts/prompts.json relative to project root
            self.prompts_file = Path(__file__).parent.parent.parent / "prompts" / "prompts.json"
        
        if not self.prompts_file.exists():
            raise ValueError(f"Prompts file not found: {self.prompts_file}")
        
        # Load all prompts
        self.prompts_data = self._load_prompts()
        
        # Extract base system prompt
        self.base_system_prompt = self.prompts_data.get("base_system_prompt", "")
        
        # Load all templates
        self.templates = self._parse_templates()
        
        logger.info(f"Loaded {len(self.templates)} prompt templates from {self.prompts_file}")
    
    def _load_prompts(self) -> Dict:
        """Load prompts from JSON file"""
        try:
            with open(self.prompts_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            raise
    
    def _parse_templates(self) -> Dict[QueryIntent, PromptTemplate]:
        """Parse templates from loaded JSON data"""
        templates = {}
        
        intents_data = self.prompts_data.get("intents", {})
        
        for intent_name, intent_data in intents_data.items():
            try:
                intent = QueryIntent(intent_name)
                
                template = PromptTemplate(
                    intent=intent,
                    system_prompt=intent_data.get("system", ""),
                    user_prompt_template=intent_data.get("user", ""),
                    response_guidelines=intent_data.get("guidelines", []),
                    example_format=intent_data.get("example_format"),
                    metadata_instructions=intent_data.get("metadata_instructions")
                )
                
                templates[intent] = template
                
            except Exception as e:
                logger.error(f"Failed to parse template for {intent_name}: {e}")
        
        # Ensure we have at least a general inquiry template
        if QueryIntent.GENERAL_INQUIRY not in templates:
            logger.warning("General inquiry template not found, creating default")
            templates[QueryIntent.GENERAL_INQUIRY] = PromptTemplate(
                intent=QueryIntent.GENERAL_INQUIRY,
                system_prompt="You are a helpful regulatory compliance assistant.",
                user_prompt_template="Query: {query}\n\nContext: {context}\n\nPlease provide a helpful response.",
                response_guidelines=["Be helpful", "Be accurate", "Cite sources"]
            )
        
        return templates
    
    def get_template(self, intent: QueryIntent) -> PromptTemplate:
        """Get template for a specific intent"""
        return self.templates.get(intent, self.templates[QueryIntent.GENERAL_INQUIRY])
    
    def build_system_prompt(self, intent: QueryIntent) -> str:
        """Build complete system prompt for an intent"""
        template = self.get_template(intent)
        
        # Combine base and specific system prompts
        full_prompt = f"{self.base_system_prompt}\n\n{template.system_prompt}"
        
        # Add response guidelines
        if template.response_guidelines:
            full_prompt += "\n\nResponse Guidelines:\n"
            for guideline in template.response_guidelines:
                full_prompt += f"- {guideline}\n"
        
        # Add example format if available
        if template.example_format:
            full_prompt += f"\n\nExample Response Format:\n{template.example_format}"
        
        return full_prompt
    
    def build_user_prompt(
        self,
        intent: QueryIntent,
        query: str,
        context: str,
        metadata: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build user prompt from template"""
        
        template = self.get_template(intent)
        
        # Prepare metadata fields
        regulatory_bodies = ", ".join(metadata.get("regulatory_bodies", ["Not specified"]))
        topics = ", ".join(list(metadata.get("topics", {}).keys())[:5])
        
        date_range = "Not specified"
        if metadata.get("date_range"):
            earliest = metadata["date_range"].get("earliest")
            latest = metadata["date_range"].get("latest")
            if earliest and latest:
                date_range = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
        
        # Format conversation history if needed
        history_str = ""
        if conversation_history and intent == QueryIntent.CLARIFICATION:
            history_str = "\n".join([
                f"{msg['role'].title()}: {msg['content']}"
                for msg in conversation_history[-3:]  # Last 3 messages
            ])
        
        # Build prompt
        prompt = template.user_prompt_template.format(
            query=query,
            context=context,
            regulatory_bodies=regulatory_bodies,
            topics=topics,
            date_range=date_range,
            conversation_history=history_str
        )
        
        # Add metadata instructions if available
        if template.metadata_instructions:
            prompt += f"\n\nAdditional context: {template.metadata_instructions}"
        
        return prompt
    
    def get_follow_up_prompt(
        self,
        original_response: str,
        follow_up_query: str,
        intent: QueryIntent
    ) -> str:
        """Create prompt for follow-up questions"""
        
        follow_up_template = self.prompts_data.get("special_prompts", {}).get("follow_up", "")
        
        return follow_up_template.format(
            original_response=original_response,
            follow_up_query=follow_up_query
        )
    
    def get_refinement_prompt(
        self,
        response: str,
        refinement_type: str = "clarity"
    ) -> str:
        """Create prompt for response refinement"""
        
        refinement_prompts = self.prompts_data.get("special_prompts", {}).get("refinement", {})
        refinement_template = refinement_prompts.get(refinement_type, refinement_prompts.get("clarity", ""))
        
        return refinement_template.format(response=response)
    
    def reload_templates(self):
        """Reload all templates from disk"""
        logger.info("Reloading prompt templates...")
        
        # Reload prompts data
        self.prompts_data = self._load_prompts()
        
        # Re-extract base system prompt
        self.base_system_prompt = self.prompts_data.get("base_system_prompt", "")
        
        # Reload all templates
        self.templates = self._parse_templates()
        
        logger.info(f"Reloaded {len(self.templates)} templates")
    
    def list_available_intents(self) -> List[str]:
        """List all available query intents"""
        return [intent.value for intent in self.templates.keys()]
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded prompts"""
        stats = {
            "prompts_file": str(self.prompts_file),
            "total_templates": len(self.templates),
            "available_intents": self.list_available_intents(),
            "base_prompt_length": len(self.base_system_prompt),
            "templates": {}
        }
        
        for intent, template in self.templates.items():
            stats["templates"][intent.value] = {
                "system_prompt_length": len(template.system_prompt),
                "user_prompt_length": len(template.user_prompt_template),
                "guidelines_count": len(template.response_guidelines),
                "has_example": template.example_format is not None,
                "has_metadata_instructions": template.metadata_instructions is not None
            }
        
        return stats