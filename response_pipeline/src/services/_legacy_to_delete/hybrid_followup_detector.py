"""
Hybrid Follow-Up Detector - Fast pattern matching with LLM fallback for context-aware Q&A
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.clients import AzureOpenAIClient, LLMRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FollowUpContext:
    """Context extracted from follow-up detection"""
    is_followup: bool
    confidence: float
    referring_entities: List[str]
    question_type: str  # continuation, clarification, expansion, comparison
    context_needed: List[str]  # What context from previous messages is needed

class HybridFollowUpDetector:
    """
    Fast follow-up detection using pattern matching with LLM fallback
    
    Performance target: <500ms average (pattern matching ~50ms, LLM fallback ~400ms)
    """
    
    def __init__(self, openai_client: Optional[AzureOpenAIClient] = None):
        self.openai_client = openai_client or AzureOpenAIClient()
        
        # Fast pattern matching rules
        self.followup_patterns = {
            # Pronouns and references
            'pronouns': [
                r'\b(it|this|that|these|those|they|them)\b',
                r'\b(such|same|similar)\b',
                r'\b(above|mentioned|previous)\b'
            ],
            
            # Question continuations
            'continuations': [
                r'\bwhat about\b',
                r'\bhow about\b',
                r'\band what\b',
                r'\balso\b.*\?',
                r'\bwhat else\b',
                r'\bany other\b',
                r'\bmore specifically\b'
            ],
            
            # Clarifications
            'clarifications': [
                r'\bcan you (clarify|explain|elaborate)\b',
                r'\bwhat (do you mean|does (this|that) mean)\b',
                r'\b(i don\'t understand|unclear|confusing)\b',
                r'\bmore details?\b'
            ],
            
            # Comparisons
            'comparisons': [
                r'\bcompared to\b',
                r'\bversus\b',
                r'\bdifference between\b',
                r'\bhow (does|do) (this|that|these|those) differ\b',
                r'\bsimilar to\b'
            ],
            
            # Expansions
            'expansions': [
                r'\bwhat are the implications\b',
                r'\bwhat happens if\b',
                r'\bconsequences\b',
                r'\bgive me examples\b',
                r'\bfor instance\b'
            ]
        }
        
        # Entity reference patterns
        self.entity_patterns = {
            'jurisdictions': r'\b(gdpr|ccpa|california|eu|europe|canada|quebec|germany|france|uk|united kingdom)\b',
            'regulations': r'\b(article \d+|section \d+|regulation|directive|law|act)\b',
            'concepts': r'\b(consent|processing|data|personal|sensitive|withdrawal|revocation)\b'
        }
        
    async def detect_followup(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]],
        max_history_messages: int = 3
    ) -> FollowUpContext:
        """
        Detect if current query is a follow-up using hybrid approach
        
        Args:
            current_query: The user's current question
            conversation_history: Recent conversation messages
            max_history_messages: Maximum previous messages to consider
            
        Returns:
            FollowUpContext with detection results
        """
        
        # Step 1: Fast pattern matching (target: <50ms)
        pattern_result = self._pattern_based_detection(current_query, conversation_history)
        
        # If pattern matching is confident (>0.8), use it
        if pattern_result.confidence > 0.8:
            logger.debug(f"Pattern-based follow-up detection: {pattern_result.confidence:.2f}")
            return pattern_result
            
        # Step 2: LLM fallback for ambiguous cases (target: <400ms)
        if len(conversation_history) > 0:
            llm_result = await self._llm_based_detection(current_query, conversation_history[-max_history_messages:])
            
            # Combine pattern and LLM insights
            combined_result = self._combine_results(pattern_result, llm_result)
            logger.debug(f"Combined follow-up detection: {combined_result.confidence:.2f}")
            return combined_result
        
        return pattern_result
    
    def _pattern_based_detection(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> FollowUpContext:
        """Fast pattern-based follow-up detection"""
        
        query_lower = current_query.lower()
        confidence = 0.0
        question_type = "standalone"
        referring_entities = []
        context_needed = []
        
        # Check for follow-up patterns
        pattern_scores = {}
        for category, patterns in self.followup_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, query_lower, re.IGNORECASE))
            if matches > 0:
                pattern_scores[category] = matches / len(patterns)
        
        # Calculate confidence based on pattern matches
        if pattern_scores:
            confidence = min(sum(pattern_scores.values()) * 0.3, 0.9)  # Max 0.9 for patterns
            question_type = max(pattern_scores.keys(), key=pattern_scores.get)
        
        # Extract entity references
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                referring_entities.extend(matches)
                context_needed.append(entity_type)
        
        # Boost confidence if query is very short (likely referring to previous context)
        words = query_lower.split()
        if len(words) < 5 and any(word in ['it', 'this', 'that', 'these', 'those'] for word in words):
            confidence = max(confidence, 0.7)
            question_type = "continuation"
        
        # Check for question words without context (likely follow-ups)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        starts_with_question = any(query_lower.startswith(word) for word in question_words)
        if starts_with_question and len(words) < 8 and not any(entity in query_lower for entity in ['gdpr', 'ccpa', 'consent']):
            confidence = max(confidence, 0.6)
            question_type = "clarification"
        
        is_followup = confidence > 0.4
        
        return FollowUpContext(
            is_followup=is_followup,
            confidence=confidence,
            referring_entities=referring_entities,
            question_type=question_type,
            context_needed=context_needed
        )
    
    async def _llm_based_detection(
        self,
        current_query: str,
        recent_history: List[Dict[str, str]]
    ) -> FollowUpContext:
        """LLM-based follow-up detection for ambiguous cases"""
        
        # Build context from recent messages
        context_messages = []
        for msg in recent_history[-3:]:  # Last 3 messages only
            role = msg.get('role', 'user')
            content = msg.get('content', '')[:200]  # Truncate for efficiency
            context_messages.append(f"{role}: {content}")
        
        context_str = "\n".join(context_messages)
        
        system_prompt = """You are a follow-up question detector for a regulatory compliance chatbot.

Analyze if the current query is a follow-up to previous messages or a standalone question.

A follow-up question:
- References previous topics without full context
- Uses pronouns (it, this, that, these, those)
- Builds on previous answers
- Asks for clarification or expansion

Return JSON with this structure:
{
  "is_followup": boolean,
  "confidence": float (0.0-1.0),
  "question_type": "continuation|clarification|expansion|comparison|standalone", 
  "referring_entities": ["entity1", "entity2"],
  "context_needed": ["previous_topic", "specific_concept"]
}

Keep analysis brief - this needs to be fast."""

        user_prompt = f"""Recent conversation:
{context_str}

Current query: {current_query}

Is this a follow-up question?"""

        messages = self.openai_client.create_messages(
            system_prompt=system_prompt,
            user_query=user_prompt,
            history=[]
        )
        
        llm_request = LLMRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.0,
            max_tokens=200,  # Keep short for speed
            stream=False,
            user="followup_detector",
            response_format={"type": "json_object"}
        )
        
        try:
            response = await self.openai_client.complete(llm_request)
            result_data = json.loads(response.content)
            
            return FollowUpContext(
                is_followup=result_data.get("is_followup", False),
                confidence=result_data.get("confidence", 0.0),
                referring_entities=result_data.get("referring_entities", []),
                question_type=result_data.get("question_type", "standalone"),
                context_needed=result_data.get("context_needed", [])
            )
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM follow-up detection failed: {e}")
            # Return low-confidence standalone result
            return FollowUpContext(
                is_followup=False,
                confidence=0.2,
                referring_entities=[],
                question_type="standalone",
                context_needed=[]
            )
    
    def _combine_results(
        self,
        pattern_result: FollowUpContext,
        llm_result: FollowUpContext
    ) -> FollowUpContext:
        """Combine pattern and LLM results intelligently"""
        
        # Average confidence with slight bias toward LLM (more context-aware)
        combined_confidence = (pattern_result.confidence * 0.4 + llm_result.confidence * 0.6)
        
        # Use LLM question type if confident, otherwise pattern
        question_type = llm_result.question_type if llm_result.confidence > 0.6 else pattern_result.question_type
        
        # Combine entities (deduplicate)
        all_entities = list(set(pattern_result.referring_entities + llm_result.referring_entities))
        
        # Combine context needed
        all_context = list(set(pattern_result.context_needed + llm_result.context_needed))
        
        # Final decision: follow-up if either method is confident
        is_followup = (combined_confidence > 0.5) or (
            pattern_result.confidence > 0.7 or llm_result.confidence > 0.7
        )
        
        return FollowUpContext(
            is_followup=is_followup,
            confidence=combined_confidence,
            referring_entities=all_entities,
            question_type=question_type,
            context_needed=all_context
        )