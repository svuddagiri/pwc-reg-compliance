"""
Sensitive Query Handler - Special handling for surveillance/criminal law queries
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SensitivityLevel(str, Enum):
    """Sensitivity levels for queries"""
    SAFE = "safe"
    MODERATE = "moderate"
    HIGH = "high"
    RETRIEVAL_ONLY = "retrieval_only"


@dataclass
class SensitiveQueryResult:
    """Result of sensitive query analysis"""
    is_sensitive: bool
    level: SensitivityLevel
    topics: List[str]
    handling_strategy: str
    neutral_rephrasing: Optional[str] = None
    requires_citation_only: bool = False


class SensitiveQueryHandler:
    """Handles queries about sensitive regulatory topics"""
    
    def __init__(self):
        # Define sensitive patterns and their handling strategies
        self.sensitive_patterns = {
            # Surveillance and interception
            "surveillance": {
                "patterns": [
                    r"intercept.*communication",
                    r"wiretap",
                    r"surveillance",
                    r"recording.*without.*consent",
                    r"listening.*calls",
                    r"§\s*2511",
                    r"one.*party.*consent"
                ],
                "level": SensitivityLevel.HIGH,
                "topics": ["surveillance", "wiretapping", "interception"],
                "strategy": "retrieval_only"
            },
            
            # Criminal penalties
            "criminal": {
                "patterns": [
                    r"prison.*term",
                    r"imprisonment",
                    r"criminal.*penalty",
                    r"jail.*time",
                    r"years.*prison",
                    r"felony",
                    r"misdemeanor"
                ],
                "level": SensitivityLevel.MODERATE,
                "topics": ["criminal_penalties", "enforcement"],
                "strategy": "neutral_summary"
            },
            
            # Electronic health records (in surveillance context)
            "ehr_surveillance": {
                "patterns": [
                    r"electronic.*health.*record.*disclos",
                    r"EHR.*accounting",
                    r"health.*record.*audit",
                    r"medical.*record.*tracking"
                ],
                "level": SensitivityLevel.MODERATE,
                "topics": ["health_records", "privacy"],
                "strategy": "compliance_focused"
            }
        }
        
        # Neutral rephrasing templates
        self.rephrase_templates = {
            "surveillance": "Summarize the regulatory requirements in {regulation} {section} based on the retrieved content",
            "criminal": "What compliance obligations and consequences are outlined in {regulation} {section}",
            "ehr_surveillance": "What are the documentation and tracking requirements for {topic} under {regulation}"
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, config in self.sensitive_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in config["patterns"]
            ]
    
    def analyze_query(self, query: str, query_analysis: Dict = None) -> SensitiveQueryResult:
        """Analyze if query is sensitive and determine handling strategy"""
        
        # Check each category
        matched_categories = []
        all_topics = []
        highest_level = SensitivityLevel.SAFE
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    matched_categories.append(category)
                    config = self.sensitive_patterns[category]
                    all_topics.extend(config["topics"])
                    
                    # Update highest sensitivity level
                    if config["level"] == SensitivityLevel.HIGH:
                        highest_level = SensitivityLevel.HIGH
                    elif config["level"] == SensitivityLevel.MODERATE and highest_level != SensitivityLevel.HIGH:
                        highest_level = SensitivityLevel.MODERATE
                    break
        
        # If no sensitive patterns found
        if not matched_categories:
            return SensitiveQueryResult(
                is_sensitive=False,
                level=SensitivityLevel.SAFE,
                topics=[],
                handling_strategy="standard",
                requires_citation_only=False
            )
        
        # Determine handling strategy
        if highest_level == SensitivityLevel.HIGH:
            strategy = "retrieval_only"
            requires_citation = True
        elif "surveillance" in matched_categories:
            strategy = "neutral_summary"
            requires_citation = True
        else:
            strategy = "compliance_focused"
            requires_citation = False
        
        # Generate neutral rephrasing
        rephrasing = self._generate_neutral_rephrasing(query, matched_categories[0], query_analysis)
        
        logger.info(
            f"Sensitive query detected: level={highest_level}, "
            f"categories={matched_categories}, strategy={strategy}"
        )
        
        return SensitiveQueryResult(
            is_sensitive=True,
            level=highest_level,
            topics=list(set(all_topics)),
            handling_strategy=strategy,
            neutral_rephrasing=rephrasing,
            requires_citation_only=requires_citation
        )
    
    def _generate_neutral_rephrasing(
        self, 
        query: str, 
        category: str,
        query_analysis: Dict = None
    ) -> str:
        """Generate a neutral rephrasing for sensitive queries"""
        
        template = self.rephrase_templates.get(category, 
            "Summarize the regulatory requirements based on the retrieved content")
        
        # Extract regulation and section if available
        regulation = "the regulation"
        section = ""
        
        if query_analysis:
            regs = query_analysis.get("search_filters", {}).get("regulation", [])
            if regs:
                regulation = regs[0]
        
        # Try to extract section numbers
        section_match = re.search(r"§\s*(\d+(?:\.\d+)?(?:\([a-z0-9]+\))?)", query)
        if section_match:
            section = f"Section {section_match.group(1)}"
        
        # Extract topic
        topic_match = re.search(r"(electronic.*?record|accounting.*?duty|disclosure)", query, re.IGNORECASE)
        topic = topic_match.group(1) if topic_match else "this topic"
        
        return template.format(
            regulation=regulation,
            section=section,
            topic=topic
        )
    
    def create_safe_prompt(self, query_result: SensitiveQueryResult, context: str) -> str:
        """Create a safe prompt for sensitive queries"""
        
        if query_result.handling_strategy == "retrieval_only":
            return f"""Based ONLY on the following retrieved regulatory text, provide a factual summary.
Do not add any interpretation or analysis beyond what is explicitly stated.

Retrieved content:
{context}

Task: {query_result.neutral_rephrasing or "Summarize the key points from the retrieved content"}"""
        
        elif query_result.handling_strategy == "neutral_summary":
            return f"""You are a regulatory compliance assistant. Provide a factual summary based solely on the retrieved content below.
Focus on compliance requirements and obligations.

Retrieved content:
{context}

Task: {query_result.neutral_rephrasing}"""
        
        else:  # compliance_focused
            return f"""As a compliance assistant, summarize the regulatory requirements from the retrieved content.
Focus on what organizations need to do to comply.

Retrieved content:
{context}

Task: Explain the compliance requirements related to {', '.join(query_result.topics)}"""
    
    def should_use_fallback_model(self, response: str) -> bool:
        """Check if response indicates model refusal"""
        
        refusal_patterns = [
            "i cannot",
            "i can't",
            "i'm unable",
            "i apologize",
            "cannot provide",
            "can't help",
            "unable to assist",
            "cannot discuss"
        ]
        
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in refusal_patterns)
    
    def extract_direct_citations(self, search_results: List[Dict], query: str) -> List[Dict]:
        """Extract direct citations for retrieval-only responses"""
        
        citations = []
        
        # Look for specific section references in search results
        for result in search_results[:5]:  # Top 5 most relevant
            # Handle both dict and DocumentChunk objects
            if hasattr(result, '__dict__'):
                # DocumentChunk object - convert to dict for processing
                chunk = result.__dict__
            elif isinstance(result, dict):
                chunk = result.get("chunk", {})
            else:
                continue
            # Extract content and metadata from chunk
            if hasattr(result, '__dict__'):
                # DocumentChunk object
                content = getattr(result, 'content', '')
                metadata = getattr(result, 'metadata', {})
                document_name = getattr(result, 'document_name', 'Unknown')
                score = getattr(result, 'score', 0.0)
            else:
                # Dict format
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                document_name = chunk.get("document_name", "Unknown")
                score = result.get("score", 0.0)
            
            # Create citation
            citation = {
                "source": document_name,
                "content": content[:500] + "..." if len(content) > 500 else content,
                "regulation": metadata.get("regulation", "Unknown") if isinstance(metadata, dict) else "Unknown",
                "section": metadata.get("clause_number", "") or metadata.get("section", "") if isinstance(metadata, dict) else "",
                "relevance_score": score
            }
            
            citations.append(citation)
        
        return citations
    
    def format_citation_only_response(self, citations: List[Dict], query: str) -> str:
        """Format a response using only citations"""
        
        if not citations:
            return "No relevant regulatory content found for this query."
        
        response = "Based on the retrieved regulatory content:\n\n"
        
        for i, citation in enumerate(citations, 1):
            response += f"**{citation['regulation']} - {citation['section']}**\n"
            response += f"{citation['content']}\n\n"
            
            if i >= 3:  # Limit to top 3 citations
                break
        
        response += "\n*Note: This response contains only direct quotes from regulatory documents.*"
        
        return response