"""
Prompt Guard - Protection against prompt injection attacks
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PromptValidationResult:
    """Result of prompt validation"""
    is_safe: bool
    threat_level: ThreatLevel
    detected_patterns: List[str]
    risk_score: float
    sanitized_prompt: Optional[str]
    reason: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "threat_level": self.threat_level.value,
            "detected_patterns": self.detected_patterns,
            "risk_score": self.risk_score,
            "reason": self.reason
        }


class PromptGuard:
    """Protects against prompt injection attacks"""
    
    def __init__(self):
        # Define injection patterns with threat levels
        self.injection_patterns = [
            # Critical threats - System prompt override attempts
            (r"ignore\s+(all\s+)?previous\s+instructions?", ThreatLevel.CRITICAL, "system_override"),
            (r"disregard\s+(all\s+)?prior\s+(instructions?|prompts?)", ThreatLevel.CRITICAL, "system_override"),
            (r"forget\s+everything\s+before", ThreatLevel.CRITICAL, "system_override"),
            (r"new\s+instructions?:\s*", ThreatLevel.CRITICAL, "system_override"),
            (r"system\s*:\s*", ThreatLevel.CRITICAL, "role_manipulation"),
            (r"assistant\s*:\s*", ThreatLevel.CRITICAL, "role_manipulation"),
            
            # High threats - Data extraction attempts
            (r"(print|show|display|reveal)\s+(all\s+)?(your\s+)?(system\s+)?prompts?", ThreatLevel.HIGH, "data_extraction"),
            (r"(show|reveal|display)\s+(the\s+)?instructions?", ThreatLevel.HIGH, "data_extraction"),
            (r"what\s+are\s+your\s+instructions?\?", ThreatLevel.HIGH, "data_extraction"),
            (r"repeat\s+the\s+above", ThreatLevel.HIGH, "data_extraction"),
            (r"print\s+everything\s+above", ThreatLevel.HIGH, "data_extraction"),
            
            # High threats - Role manipulation
            (r"you\s+are\s+now\s+", ThreatLevel.HIGH, "role_manipulation"),
            (r"act\s+as\s+(a|an)\s+", ThreatLevel.HIGH, "role_manipulation"),
            (r"pretend\s+(to\s+be|you\s+are)", ThreatLevel.HIGH, "role_manipulation"),
            (r"from\s+now\s+on\s+you", ThreatLevel.HIGH, "role_manipulation"),
            
            # Medium threats - Encoding attempts
            (r"base64\s*:\s*[A-Za-z0-9+/=]{20,}", ThreatLevel.MEDIUM, "encoding_attempt"),
            (r"\\x[0-9a-fA-F]{2}", ThreatLevel.MEDIUM, "encoding_attempt"),
            (r"\\u[0-9a-fA-F]{4}", ThreatLevel.MEDIUM, "encoding_attempt"),
            
            # Medium threats - Jailbreak attempts
            (r"DAN\s+mode", ThreatLevel.MEDIUM, "jailbreak"),
            (r"developer\s+mode", ThreatLevel.MEDIUM, "jailbreak"),
            (r"enable\s+.*\s+mode", ThreatLevel.MEDIUM, "jailbreak"),
            
            # Low threats - Suspicious patterns
            (r"</?script[^>]*>", ThreatLevel.LOW, "code_injection"),
            (r"javascript\s*:", ThreatLevel.LOW, "code_injection"),
            (r"eval\s*\(", ThreatLevel.LOW, "code_injection"),
        ]
        
        # Additional validation rules
        self.max_prompt_length = 4000
        self.max_url_count = 3
        self.max_code_block_count = 2
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), level, category) 
            for pattern, level, category in self.injection_patterns
        ]
        
    async def validate_prompt(self, prompt: str, user_context: Optional[Dict] = None) -> PromptValidationResult:
        """Validate a prompt for security threats"""
        
        if not prompt:
            return PromptValidationResult(
                is_safe=False,
                threat_level=ThreatLevel.LOW,
                detected_patterns=[],
                risk_score=0.3,
                sanitized_prompt=None,
                reason="Empty prompt"
            )
        
        # Check length
        if len(prompt) > self.max_prompt_length:
            return PromptValidationResult(
                is_safe=False,
                threat_level=ThreatLevel.MEDIUM,
                detected_patterns=["excessive_length"],
                risk_score=0.6,
                sanitized_prompt=prompt[:self.max_prompt_length],
                reason=f"Prompt exceeds maximum length of {self.max_prompt_length} characters"
            )
        
        # Check for injection patterns
        detected_patterns = []
        highest_threat = ThreatLevel.NONE
        risk_score = 0.0
        
        for pattern, threat_level, category in self.compiled_patterns:
            if pattern.search(prompt):
                detected_patterns.append(f"{category}:{pattern.pattern[:30]}...")
                # Update highest threat level
                if self._threat_level_value(threat_level) > self._threat_level_value(highest_threat):
                    highest_threat = threat_level
                # Accumulate risk score
                risk_score += self._threat_level_value(threat_level) * 0.25
        
        # Check for other suspicious patterns
        url_count = len(re.findall(r'https?://[^\s]+', prompt))
        if url_count > self.max_url_count:
            detected_patterns.append(f"excessive_urls:{url_count}")
            risk_score += 0.2
            if highest_threat == ThreatLevel.NONE:
                highest_threat = ThreatLevel.LOW
        
        # Check for code blocks
        code_block_count = len(re.findall(r'```[\s\S]*?```', prompt))
        if code_block_count > self.max_code_block_count:
            detected_patterns.append(f"excessive_code_blocks:{code_block_count}")
            risk_score += 0.1
            if highest_threat == ThreatLevel.NONE:
                highest_threat = ThreatLevel.LOW
        
        # Check for repeated characters (potential DoS)
        if self._has_excessive_repetition(prompt):
            detected_patterns.append("excessive_repetition")
            risk_score += 0.3
            if highest_threat in [ThreatLevel.NONE, ThreatLevel.LOW]:
                highest_threat = ThreatLevel.MEDIUM
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine if safe
        is_safe = highest_threat in [ThreatLevel.NONE, ThreatLevel.LOW] and risk_score < 0.5
        
        # Sanitize if needed
        sanitized_prompt = None
        if not is_safe and highest_threat != ThreatLevel.CRITICAL:
            sanitized_prompt = self._sanitize_prompt(prompt)
        
        # Build reason
        reason = None
        if detected_patterns:
            reason = f"Detected patterns: {', '.join(detected_patterns[:3])}"
            if len(detected_patterns) > 3:
                reason += f" and {len(detected_patterns) - 3} more"
        
        result = PromptValidationResult(
            is_safe=is_safe,
            threat_level=highest_threat,
            detected_patterns=detected_patterns,
            risk_score=risk_score,
            sanitized_prompt=sanitized_prompt,
            reason=reason
        )
        
        # Log high-risk attempts
        if highest_threat in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.warning(
                f"High-risk prompt detected: threat_level={highest_threat.value}, "
                f"patterns={detected_patterns[:3]}, risk_score={risk_score:.2f}"
            )
        
        return result
    
    def _threat_level_value(self, level: ThreatLevel) -> float:
        """Convert threat level to numeric value"""
        return {
            ThreatLevel.NONE: 0.0,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 1.0
        }.get(level, 0.0)
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character repetition"""
        # Check for repeated characters (e.g., "aaaaaaa")
        if re.search(r'(.)\1{9,}', text):
            return True
        # Check for repeated words
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
                if word_counts[word_lower] > len(words) * 0.3:  # More than 30% repetition
                    return True
        return False
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt by removing dangerous patterns"""
        sanitized = prompt
        
        # Remove obvious injection attempts
        dangerous_patterns = [
            r"ignore\s+.*\s+instructions?",
            r"system\s*:\s*",
            r"assistant\s*:\s*",
            r"</?script[^>]*>",
            r"javascript\s*:",
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
        
        # Remove excessive repetition
        sanitized = re.sub(r'(.)\1{9,}', r'\1\1\1...', sanitized)
        
        return sanitized
    
    def add_custom_pattern(self, pattern: str, threat_level: ThreatLevel, category: str):
        """Add a custom injection pattern"""
        self.injection_patterns.append((pattern, threat_level, category))
        self.compiled_patterns.append((re.compile(pattern, re.IGNORECASE), threat_level, category))
        logger.info(f"Added custom pattern: {category} with threat level {threat_level.value}")