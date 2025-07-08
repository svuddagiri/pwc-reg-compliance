"""
Content Filter - Pre and post filtering for inappropriate content
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContentType(str, Enum):
    """Types of content issues"""
    CLEAN = "clean"
    PII = "pii"
    PROFANITY = "profanity"
    HARMFUL = "harmful"
    BIAS = "bias"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"


@dataclass
class ContentFilterResult:
    """Result of content filtering"""
    is_safe: bool
    content_types: List[ContentType]
    filtered_content: Optional[str]
    redacted_items: List[Dict[str, str]]
    severity: str  # low, medium, high
    reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "content_types": [ct.value for ct in self.content_types],
            "redacted_items": self.redacted_items,
            "severity": self.severity,
            "reason": self.reason
        }


class ContentFilter:
    """Filters inappropriate or harmful content"""
    
    def __init__(self):
        # PII patterns
        self.pii_patterns = {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            "passport": re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
            "drivers_license": re.compile(r'\b[A-Z]{1,2}\d{5,8}\b'),
        }
        
        # Sensitive topics that need careful handling
        self.sensitive_topics = {
            "medical": ["diagnosis", "prescription", "treatment", "medical advice", "symptoms"],
            "legal": ["legal advice", "lawsuit", "attorney", "court", "legal opinion"],
            "financial": ["investment advice", "financial planning", "tax advice", "trading"],
        }
        
        # Harmful content patterns
        self.harmful_patterns = [
            r'\b(suicide|self.?harm|kill\s+yourself)\b',
            r'\b(make\s+bomb|build\s+weapon|create\s+explosive)\b',
            r'\b(hack|breach|exploit)\s+.{0,20}(system|network|database)\b',
        ]
        
        # System prompt leakage patterns (for post-filtering)
        self.system_prompt_patterns = [
            r'You\s+are\s+a\s+regulatory\s+compliance\s+expert',
            r'Important\s+guidelines:',
            r'Only\s+use\s+information\s+from\s+the\s+provided\s+context',
            r'System\s+prompt:',
            r'Instructions:.*Do\s+not\s+reveal',
        ]
        
        # Compile patterns
        self.compiled_harmful = [re.compile(p, re.IGNORECASE) for p in self.harmful_patterns]
        self.compiled_system = [re.compile(p, re.IGNORECASE) for p in self.system_prompt_patterns]
    
    async def pre_filter(self, content: str) -> ContentFilterResult:
        """Filter content before sending to LLM"""
        
        detected_types = []
        redacted_items = []
        filtered_content = content
        severity = "low"
        
        # Check for PII
        pii_found = self._check_pii(content)
        if pii_found:
            detected_types.append(ContentType.PII)
            filtered_content, pii_redactions = self._redact_pii(filtered_content)
            redacted_items.extend(pii_redactions)
            severity = "medium"
        
        # Check for harmful content
        harmful_found = self._check_harmful_content(content)
        if harmful_found:
            detected_types.append(ContentType.HARMFUL)
            severity = "high"
        
        # Check for sensitive topics
        sensitive_topics = self._check_sensitive_topics(content)
        detected_types.extend(sensitive_topics)
        
        # Determine if safe
        is_safe = ContentType.HARMFUL not in detected_types
        
        # Build reason
        reason = None
        if detected_types:
            reason = f"Detected: {', '.join([ct.value for ct in detected_types])}"
        
        if not detected_types:
            detected_types = [ContentType.CLEAN]
        
        return ContentFilterResult(
            is_safe=is_safe,
            content_types=detected_types,
            filtered_content=filtered_content if is_safe else None,
            redacted_items=redacted_items,
            severity=severity,
            reason=reason
        )
    
    async def post_filter(self, response: str) -> ContentFilterResult:
        """Filter LLM response before returning to user"""
        
        print(f"DEBUG POST_FILTER: Input response length: {len(response)} chars")
        print(f"DEBUG POST_FILTER: Input starts with: {response[:100]}...")
        
        detected_types = []
        redacted_items = []
        filtered_response = response
        severity = "low"
        
        # Check for system prompt leakage
        system_leaks = self._check_system_prompt_leakage(response)
        if system_leaks:
            detected_types.append(ContentType.HARMFUL)
            filtered_response = self._remove_system_prompts(filtered_response)
            redacted_items.extend([{"type": "system_prompt", "text": leak} for leak in system_leaks])
            severity = "high"
            logger.warning(f"System prompt leakage detected: {len(system_leaks)} instances")
        
        # Check for PII in response
        pii_found = self._check_pii(response)
        if pii_found:
            detected_types.append(ContentType.PII)
            filtered_response, pii_redactions = self._redact_pii(filtered_response)
            redacted_items.extend(pii_redactions)
            if severity == "low":
                severity = "medium"
        
        # Check for harmful content in response
        harmful_found = self._check_harmful_content(response)
        if harmful_found:
            detected_types.append(ContentType.HARMFUL)
            severity = "high"
        
        # Validate citations are not fabricated
        if not self._validate_citations(response):
            # Don't mark as unsafe, but log for monitoring
            logger.warning("Potentially fabricated citations detected in response")
        
        # Determine if safe
        is_safe = ContentType.HARMFUL not in detected_types
        
        # Build reason
        reason = None
        if detected_types:
            reason = f"Response filtered for: {', '.join([ct.value for ct in detected_types])}"
        
        if not detected_types:
            detected_types = [ContentType.CLEAN]
        
        return ContentFilterResult(
            is_safe=is_safe,
            content_types=detected_types,
            filtered_content=filtered_response if is_safe else None,
            redacted_items=redacted_items,
            severity=severity,
            reason=reason
        )
    
    def _check_pii(self, content: str) -> bool:
        """Check if content contains PII"""
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(content):
                return True
        return False
    
    def _redact_pii(self, content: str) -> tuple[str, List[Dict[str, str]]]:
        """Redact PII from content"""
        redacted = content
        redactions = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                redacted = redacted.replace(match, f"[{pii_type.upper()}_REDACTED]")
                redactions.append({"type": pii_type, "text": match[:4] + "..." if len(match) > 4 else match})
        
        return redacted, redactions
    
    def _check_harmful_content(self, content: str) -> bool:
        """Check for harmful content patterns"""
        for pattern in self.compiled_harmful:
            if pattern.search(content):
                return True
        return False
    
    def _check_sensitive_topics(self, content: str) -> List[ContentType]:
        """Check for sensitive topics that need disclaimers"""
        found_topics = []
        content_lower = content.lower()
        
        for topic, keywords in self.sensitive_topics.items():
            for keyword in keywords:
                if keyword in content_lower:
                    if topic == "medical":
                        found_topics.append(ContentType.MEDICAL)
                    elif topic == "legal":
                        found_topics.append(ContentType.LEGAL)
                    elif topic == "financial":
                        found_topics.append(ContentType.FINANCIAL)
                    break
        
        return list(set(found_topics))  # Remove duplicates
    
    def _check_system_prompt_leakage(self, response: str) -> List[str]:
        """Check if response contains system prompt leakage"""
        leaks = []
        
        for pattern in self.compiled_system:
            matches = pattern.findall(response)
            leaks.extend(matches)
        
        return leaks
    
    def _remove_system_prompts(self, response: str) -> str:
        """Remove system prompt leakage from response"""
        filtered = response
        
        for pattern in self.compiled_system:
            filtered = pattern.sub("[SYSTEM_CONTENT_REMOVED]", filtered)
        
        return filtered
    
    def _validate_citations(self, response: str) -> bool:
        """Basic validation that citations follow expected format"""
        # Look for citation pattern [Doc: X, Clause: Y]
        citation_pattern = re.compile(r'\[Doc:\s*([^,\]]+),\s*Clause:\s*([^\]]+)\]')
        citations = citation_pattern.findall(response)
        
        # If response has citations, ensure they're not obviously fake
        for doc, clause in citations:
            # Check for suspicious patterns
            if any(suspicious in doc.lower() for suspicious in ['example', 'test', 'fake', 'sample']):
                return False
            if clause.strip() == '' or clause.lower() == 'n/a':
                return False
        
        return True
    
    def add_blocked_term(self, term: str, content_type: ContentType = ContentType.HARMFUL):
        """Add a custom blocked term"""
        pattern = rf'\b{re.escape(term)}\b'
        self.harmful_patterns.append(pattern)
        self.compiled_harmful.append(re.compile(pattern, re.IGNORECASE))
        logger.info(f"Added blocked term: {term} as {content_type.value}")