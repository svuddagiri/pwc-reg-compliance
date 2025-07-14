"""
Fast Query Expander - Template-based query expansion for context-aware follow-up questions
"""
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.services.hybrid_followup_detector import FollowUpContext
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ExpandedQuery:
    """Result of query expansion"""
    original_query: str
    expanded_query: str
    context_entities: Dict[str, List[str]]  # jurisdictions, regulations, concepts
    expansion_method: str  # template, context_injection, hybrid
    confidence: float

class FastQueryExpander:
    """
    Fast query expansion using templates and context injection (no LLM calls)
    
    Performance target: <100ms (template matching + string operations)
    """
    
    def __init__(self):
        # Query expansion templates for common follow-up patterns
        self.expansion_templates = {
            'pronouns': {
                'it': 'the {last_concept}',
                'this': 'the {last_concept}',
                'that': 'the {last_regulation} requirement',
                'these': 'the {last_concepts}',
                'those': 'the {last_regulations}',
                'they': 'the {last_jurisdictions}'
            },
            
            'continuation_starters': {
                'what about': 'What about {last_concept} in',
                'how about': 'How about {last_concept} regarding',
                'and what': 'And what {query_remainder} for {last_concept}',
                'also': 'Also, regarding {last_concept},',
                'what else': 'What else about {last_concept}',
                'any other': 'Any other {last_concept_type}'
            },
            
            'clarification_starters': {
                'can you clarify': 'Can you clarify {last_concept}',
                'what do you mean': 'What do you mean by {last_concept}',
                'more details': 'More details about {last_concept}',
                'unclear': 'Please clarify {last_concept}',
                'explain': 'Please explain {last_concept}'
            }
        }
        
        # Entity extraction patterns for context building
        self.entity_patterns = {
            'jurisdictions': [
                r'\b(gdpr|european union|eu)\b',
                r'\b(ccpa|california)\b',
                r'\b(canada|quebec|pipeda)\b',
                r'\b(germany|france|uk|united kingdom|britain)\b',
                r'\b(denmark|estonia|netherlands|italy|spain)\b',
                r'\b(costa rica|gabon|brazil)\b'
            ],
            'regulations': [
                r'\b(article \d+)\b',
                r'\b(section \d+)\b',
                r'\b(gdpr article \d+)\b',
                r'\b(regulation \d+)\b',
                r'\b(directive \d+/\d+)\b'
            ],
            'concepts': [
                r'\b(consent|explicit consent|express consent|affirmative consent)\b',
                r'\b(processing|data processing|personal data processing)\b',
                r'\b(withdrawal|revocation|opt-out)\b',
                r'\b(sensitive data|special categories|personal data)\b',
                r'\b(data subject|individual|person)\b',
                r'\b(controller|processor|organization)\b',
                r'\b(transfer|cross-border|international)\b'
            ]
        }
        
        # Question type templates
        self.question_templates = {
            'definition': "What is {concept} under {jurisdiction}?",
            'requirement': "What are the requirements for {concept} under {jurisdiction}?",
            'comparison': "How does {concept} differ between {jurisdiction1} and {jurisdiction2}?",
            'procedure': "What is the procedure for {concept} under {jurisdiction}?",
            'consequences': "What are the consequences of {concept} under {jurisdiction}?",
            'exceptions': "What are the exceptions to {concept} under {jurisdiction}?"
        }
    
    def expand_query(
        self,
        current_query: str,
        followup_context: FollowUpContext,
        conversation_context: Dict[str, any]
    ) -> ExpandedQuery:
        """
        Expand query using fast template-based approach
        
        Args:
            current_query: User's current question
            followup_context: Context from follow-up detection
            conversation_context: Previous conversation entities and topics
            
        Returns:
            ExpandedQuery with expanded query and context
        """
        
        if not followup_context.is_followup:
            # Not a follow-up, return as-is
            return ExpandedQuery(
                original_query=current_query,
                expanded_query=current_query,
                context_entities=self._extract_entities(current_query),
                expansion_method="none",
                confidence=1.0
            )
        
        # Extract context entities from conversation
        context_entities = conversation_context.get('entities', {})
        last_concepts = context_entities.get('concepts', [])
        last_jurisdictions = context_entities.get('jurisdictions', [])
        last_regulations = context_entities.get('regulations', [])
        
        # Determine expansion method based on follow-up type
        if followup_context.question_type == "continuation":
            expanded_query = self._expand_continuation(current_query, context_entities)
        elif followup_context.question_type == "clarification":
            expanded_query = self._expand_clarification(current_query, context_entities)
        elif followup_context.question_type == "comparison":
            expanded_query = self._expand_comparison(current_query, context_entities)
        elif followup_context.question_type == "expansion":
            expanded_query = self._expand_expansion(current_query, context_entities, conversation_context)
        else:
            # Fallback: simple pronoun replacement
            expanded_query = self._expand_pronouns(current_query, context_entities)
        
        # Extract entities from expanded query
        expanded_entities = self._extract_entities(expanded_query)
        
        # Merge with conversation context entities
        merged_entities = self._merge_entities(expanded_entities, context_entities)
        
        return ExpandedQuery(
            original_query=current_query,
            expanded_query=expanded_query,
            context_entities=merged_entities,
            expansion_method="template",
            confidence=followup_context.confidence
        )
    
    def _expand_continuation(self, query: str, context_entities: Dict[str, List[str]]) -> str:
        """Expand continuation questions like 'what about...', 'how about...'"""
        
        query_lower = query.lower()
        last_concept = context_entities.get('concepts', ['this topic'])[0] if context_entities.get('concepts') else 'this topic'
        last_jurisdiction = context_entities.get('jurisdictions', ['this jurisdiction'])[0] if context_entities.get('jurisdictions') else 'this jurisdiction'
        
        # Get previous queries to understand context better
        previous_queries = context_entities.get('previous_queries', [])
        last_query = previous_queries[0] if previous_queries else ''
        
        # Handle "what about X" patterns
        if query_lower.startswith('what about'):
            remainder = query[10:].strip()  # Remove "what about "
            if remainder:
                # Check if it's asking about a jurisdiction
                if self._is_jurisdiction(remainder):
                    return self._reconstruct_query_for_jurisdiction(last_query, remainder)
                else:
                    return f"What about {remainder} regarding {last_concept}"
            else:
                return f"What about {last_concept} in {last_jurisdiction}"
        
        # Handle "how about X" patterns  
        if query_lower.startswith('how about'):
            remainder = query[9:].strip()  # Remove "how about "
            if remainder:
                # Check if it's asking about a jurisdiction
                if self._is_jurisdiction(remainder):
                    return self._reconstruct_query_for_jurisdiction(last_query, remainder)
                else:
                    return f"How does {remainder} work with {last_concept}"
            else:
                return f"How does {last_concept} work in {last_jurisdiction}"
        
        # Handle "and what..." patterns
        if query_lower.startswith('and what'):
            remainder = query[8:].strip()  # Remove "and what "
            return f"And what {remainder} for {last_concept}"
        
        # Handle "also" patterns
        if query_lower.startswith('also'):
            remainder = query[4:].strip()
            return f"Also, regarding {last_concept}, {remainder}"
        
        return query  # Fallback
    
    def _is_jurisdiction(self, text: str) -> bool:
        """Check if the text is a jurisdiction name"""
        text_lower = text.lower().strip()
        # Common jurisdiction patterns
        jurisdiction_keywords = [
            'gdpr', 'eu', 'european union', 'costa rica', 'denmark', 'estonia', 
            'gabon', 'georgia', 'missouri', 'iceland', 'alabama', 'netherlands',
            'italy', 'spain', 'france', 'germany', 'brazil', 'canada', 'quebec',
            'california', 'ccpa', 'united states', 'us', 'uk', 'united kingdom'
        ]
        
        # Check if it starts with 'in' (e.g., "in Costa Rica")
        if text_lower.startswith('in '):
            text_lower = text_lower[3:]
        
        return any(jurisdiction in text_lower for jurisdiction in jurisdiction_keywords)
    
    def _reconstruct_query_for_jurisdiction(self, previous_query: str, new_jurisdiction: str) -> str:
        """Reconstruct the previous query but for a different jurisdiction"""
        # Remove 'in' if present
        if new_jurisdiction.lower().startswith('in '):
            new_jurisdiction = new_jurisdiction[3:]
        
        previous_lower = previous_query.lower()
        
        # Try to identify the pattern and reconstruct
        if 'requirements for' in previous_lower:
            # Extract what comes after "requirements for"
            parts = previous_lower.split('requirements for')
            if len(parts) > 1:
                concept_part = parts[1]
                # Remove jurisdiction references
                for jur in ['estonia', 'denmark', 'costa rica', 'gabon', 'iceland', 'missouri', 'alabama', 'georgia']:
                    concept_part = concept_part.replace(f' in {jur}', '').replace(f' for {jur}', '')
                return f"What are the requirements for{concept_part.strip()} in {new_jurisdiction}"
        
        # For other patterns, try to replace the jurisdiction
        if ' in estonia' in previous_lower:
            return previous_query.replace('Estonia', new_jurisdiction).replace('estonia', new_jurisdiction)
        
        # Fallback: construct a generic query
        if 'consent' in previous_lower:
            return f"What are the requirements for valid consent for data processing in {new_jurisdiction}"
        
        return f"What are the requirements in {new_jurisdiction}"
    
    def _expand_clarification(self, query: str, context_entities: Dict[str, List[str]]) -> str:
        """Expand clarification questions"""
        
        query_lower = query.lower()
        last_concept = context_entities.get('concepts', ['the previous topic'])[0] if context_entities.get('concepts') else 'the previous topic'
        
        clarification_patterns = {
            'can you clarify': f"Can you clarify {last_concept}",
            'what do you mean': f"What do you mean by {last_concept}",
            'more details': f"More details about {last_concept}",
            'unclear': f"Please clarify {last_concept}",
            'explain': f"Please explain {last_concept}",
            'what does': f"What does {last_concept} mean"
        }
        
        for pattern, replacement in clarification_patterns.items():
            if pattern in query_lower:
                return replacement
        
        return query  # Fallback
    
    def _expand_comparison(self, query: str, context_entities: Dict[str, List[str]]) -> str:
        """Expand comparison questions"""
        
        query_lower = query.lower()
        concepts = context_entities.get('concepts', [])
        jurisdictions = context_entities.get('jurisdictions', [])
        
        # Handle "compared to" patterns
        if 'compared to' in query_lower:
            if concepts:
                return f"How does {concepts[0]} compare between jurisdictions mentioned"
            return query
        
        # Handle "difference" patterns
        if 'difference' in query_lower:
            if concepts and len(jurisdictions) >= 2:
                return f"What is the difference in {concepts[0]} between {jurisdictions[0]} and {jurisdictions[1]}"
            elif concepts:
                return f"What are the differences in {concepts[0]} between jurisdictions"
        
        return query
    
    def _expand_expansion(self, query: str, context_entities: Dict[str, List[str]], conversation_context: Dict[str, any] = None) -> str:
        """Expand expansion questions asking for more detail"""
        
        query_lower = query.lower()
        last_concept = context_entities.get('concepts', ['this topic'])[0] if context_entities.get('concepts') else 'this topic'
        last_jurisdiction = context_entities.get('jurisdictions', ['this jurisdiction'])[0] if context_entities.get('jurisdictions') else 'this jurisdiction'
        
        # Get the most recent topic from conversation context
        last_query = ''
        if conversation_context and conversation_context.get('previous_queries'):
            last_query = conversation_context['previous_queries'][0]
        
        expansion_patterns = {
            'what are the implications': f"What are the implications of {last_concept} under {last_jurisdiction}",
            'what happens if': f"What happens if {last_concept} requirements are not met under {last_jurisdiction}",
            'consequences': f"What are the consequences of {last_concept} violations under {last_jurisdiction}",
            'give me examples': f"Give me examples of {last_concept} under {last_jurisdiction}",
            'for instance': f"For instance, how does {last_concept} work under {last_jurisdiction}",
            # Add common "more" patterns
            'give me more': self._expand_more_pattern(query_lower, last_query, last_concept),
            'couple more': self._expand_more_pattern(query_lower, last_query, last_concept),
            'few more': self._expand_more_pattern(query_lower, last_query, last_concept),
            'some more': self._expand_more_pattern(query_lower, last_query, last_concept),
            'any more': self._expand_more_pattern(query_lower, last_query, last_concept),
            'what else': f"What else about {last_concept}",
            'any other': f"Any other {last_concept}",
            'more examples': f"More examples of {last_concept}",
            'additional': f"Additional information about {last_concept}"
        }
        
        for pattern, replacement in expansion_patterns.items():
            if pattern in query_lower:
                return replacement
        
        return query
    
    def _expand_more_pattern(self, query_lower: str, last_query: str, last_concept: str) -> str:
        """Handle 'give me more', 'couple more' etc. patterns intelligently"""
        
        # Try to infer what "more" refers to from the last query
        last_query_lower = last_query.lower()
        
        # Look for patterns in the previous query to understand what "more" means
        if 'errors' in last_query_lower or 'mistakes' in last_query_lower:
            if 'couple more' in query_lower or 'few more' in query_lower:
                return f"Give me couple more errors organizations make about {last_concept}"
            else:
                return f"Give me more errors organizations make about {last_concept}"
        
        elif 'examples' in last_query_lower:
            return f"Give me more examples of {last_concept}"
        
        elif 'requirements' in last_query_lower:
            return f"Give me more requirements for {last_concept}"
        
        elif 'consequences' in last_query_lower or 'penalties' in last_query_lower:
            return f"Give me more consequences of {last_concept} violations"
        
        elif 'jurisdictions' in last_query_lower:
            return f"Give me more jurisdictions that require {last_concept}"
        
        elif 'procedures' in last_query_lower or 'steps' in last_query_lower:
            return f"Give me more details about {last_concept} procedures"
        
        # Fallback: generic expansion
        if 'couple more' in query_lower:
            return f"Give me couple more details about {last_concept}"
        else:
            return f"Give me more information about {last_concept}"
    
    def _expand_pronouns(self, query: str, context_entities: Dict[str, List[str]]) -> str:
        """Simple pronoun replacement fallback"""
        
        last_concept = context_entities.get('concepts', ['this topic'])[0] if context_entities.get('concepts') else 'this topic'
        last_jurisdiction = context_entities.get('jurisdictions', ['this jurisdiction'])[0] if context_entities.get('jurisdictions') else 'this jurisdiction'
        
        # Simple pronoun replacements
        replacements = {
            r'\bit\b': last_concept,
            r'\bthis\b': f"this {last_concept}",
            r'\bthat\b': f"that {last_concept}",
            r'\bthese\b': f"these {last_concept} requirements",
            r'\bthose\b': f"those {last_concept} requirements"
        }
        
        expanded = query
        for pattern, replacement in replacements.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using regex patterns"""
        
        entities = {
            'jurisdictions': [],
            'regulations': [],
            'concepts': []
        }
        
        text_lower = text.lower()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    entities[entity_type].extend(matches)
        
        # Deduplicate and clean
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _merge_entities(
        self, 
        new_entities: Dict[str, List[str]], 
        context_entities: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Merge new entities with context entities"""
        
        merged = {}
        for entity_type in ['jurisdictions', 'regulations', 'concepts']:
            new_list = new_entities.get(entity_type, [])
            context_list = context_entities.get(entity_type, [])
            
            # Combine and deduplicate, keeping order (new first)
            combined = new_list + [item for item in context_list if item not in new_list]
            merged[entity_type] = combined[:5]  # Keep top 5 for performance
        
        return merged