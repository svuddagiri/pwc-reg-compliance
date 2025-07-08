"""
Configuration Service - Centralized configuration management for consent-focused queries
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchFilter:
    """Represents a search filter configuration"""
    domain_filter: str
    subdomain_filter: str
    expected_chunks: int
    min_results: int
    max_results: int


@dataclass
class ScopeCheck:
    """Result of scope boundary check"""
    is_in_scope: bool
    confidence: float
    detected_topic: Optional[str]
    redirect_message: Optional[str]
    suggested_questions: List[str]


class ConfigService:
    """
    Centralized service for loading and managing configurations.
    Handles profile-based filtering, scope boundaries, and mapping lookups.
    """
    
    def __init__(self, config_root: Optional[str] = None):
        """
        Initialize ConfigService with configuration root directory.
        
        Args:
            config_root: Path to config directory. Defaults to project root /config
        """
        if config_root is None:
            # Find project root (parent of src)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            config_root = project_root / "config"
        
        self.config_root = Path(config_root)
        
        # Load all configurations
        self._load_configurations()
        
        logger.info(f"ConfigService initialized with profile: {self.active_profile_name}")
    
    def _load_configurations(self):
        """Load all configuration files"""
        try:
            # Load active profile
            with open(self.config_root / "active_profile.json", "r") as f:
                active_profile_data = json.load(f)
                self.active_profile_name = active_profile_data["profile"]
            
            # Load profile configuration
            with open(self.config_root / "profiles" / f"{self.active_profile_name}.json", "r") as f:
                self.profile = json.load(f)
            
            # Load filters
            filter_name = self.profile["filters"]["search_filters"].get("filter_name", "consent_only")
            with open(self.config_root / "filters" / f"{filter_name}.json", "r") as f:
                self.filters = json.load(f)
            
            # Load scope boundaries
            with open(self.config_root / "boundaries" / "scope_boundaries.json", "r") as f:
                self.boundaries = json.load(f)
            
            # Load mappings
            self.mappings = {}
            mappings_dir = self.config_root / "mappings"
            for mapping_file in mappings_dir.glob("*.json"):
                mapping_name = mapping_file.stem
                with open(mapping_file, "r") as f:
                    self.mappings[mapping_name] = json.load(f)
            
            logger.debug(f"Loaded configurations: profile={self.active_profile_name}, "
                        f"mappings={list(self.mappings.keys())}")
                        
        except Exception as e:
            logger.error(f"Failed to load configurations: {str(e)}")
            raise
    
    @lru_cache(maxsize=1)
    def get_search_filters(self) -> SearchFilter:
        """
        Get Azure Search filters for current profile.
        
        Returns:
            SearchFilter object with domain, subdomain filters and validation params
        """
        profile_filters = self.profile["filters"]["search_filters"]
        filter_config = self.filters
        
        # Build filter expressions
        domain_values = profile_filters["clause_domain"]["values"]
        subdomain_values = profile_filters["clause_subdomain"]["values"]
        
        # Use the recommended filter option from consent_filters.json
        azure_filter = filter_config["azure_search_filters"]["filter_expression"][
            filter_config["azure_search_filters"]["filter_expression"]["recommended"]
        ]
        
        return SearchFilter(
            domain_filter=f"clause_domain/any(d: search.in(d, '{','.join(domain_values)}'))",
            subdomain_filter=f"clause_subdomain/any(s: search.in(s, '{','.join(subdomain_values)}'))",
            expected_chunks=filter_config["expected_results"]["total_chunks"],
            min_results=filter_config["validation"]["min_results"],
            max_results=filter_config["validation"]["max_results"]
        )
    
    def get_azure_search_filter(self) -> str:
        """
        Get the complete Azure Search filter string.
        
        Returns:
            Complete filter string for Azure Search query
        """
        # Get the text search configuration
        text_search = self.filters["azure_search_filters"]["text_search"]
        
        # Return the recommended filter expression
        filter_expr = self.filters["azure_search_filters"]["filter_expression"]
        return filter_expr[filter_expr["recommended"]]
    
    def check_scope_boundary(self, query: str) -> ScopeCheck:
        """
        Check if a query is within scope boundaries.
        
        Args:
            query: User's query text
            
        Returns:
            ScopeCheck object with scope determination and redirect info
        """
        query_lower = query.lower()
        
        # Check in-scope patterns and concepts
        in_scope_concepts = self.boundaries["in_scope"]["concepts"]
        in_scope_patterns = self.boundaries["in_scope"]["question_patterns"]
        
        # Check for in-scope match
        for concept in in_scope_concepts:
            if concept.lower() in query_lower:
                return ScopeCheck(
                    is_in_scope=True,
                    confidence=0.9,
                    detected_topic=concept,
                    redirect_message=None,
                    suggested_questions=[]
                )
        
        # Check regex patterns
        for pattern in in_scope_patterns:
            if re.search(pattern, query_lower):
                return ScopeCheck(
                    is_in_scope=True,
                    confidence=0.85,
                    detected_topic="consent-related",
                    redirect_message=None,
                    suggested_questions=[]
                )
        
        # Check out-of-scope concepts
        out_of_scope_concepts = self.boundaries["out_of_scope"]["concepts"]
        detected_out_topic = None
        
        for concept in out_of_scope_concepts:
            if concept.lower() in query_lower:
                detected_out_topic = concept
                break
        
        if detected_out_topic:
            redirect_msg = self.boundaries["out_of_scope"]["redirect_message"].format(
                detected_topic=detected_out_topic
            )
            suggestions = self.boundaries["out_of_scope"]["suggest_alternatives"]
            
            return ScopeCheck(
                is_in_scope=False,
                confidence=0.8,
                detected_topic=detected_out_topic,
                redirect_message=redirect_msg,
                suggested_questions=suggestions
            )
        
        # Uncertain - low confidence
        return ScopeCheck(
            is_in_scope=False,
            confidence=0.3,
            detected_topic=None,
            redirect_message=self.boundaries["confidence_handling"]["low_confidence"]["response"],
            suggested_questions=self.boundaries["out_of_scope"]["suggest_alternatives"]
        )
    
    def get_confidence_response(self, confidence: float, detected_topic: Optional[str] = None) -> Optional[str]:
        """
        Get appropriate response based on confidence level.
        
        Args:
            confidence: Confidence score (0-1)
            detected_topic: Optional detected topic
            
        Returns:
            Appropriate response message or None if high confidence
        """
        confidence_config = self.boundaries["confidence_handling"]
        
        if confidence >= confidence_config["high_confidence"]["threshold"]:
            return None  # Proceed normally
        elif confidence >= confidence_config["medium_confidence"]["threshold"]:
            template = confidence_config["medium_confidence"]["response"]
            intent = detected_topic or "your question"
            return template.format(detected_intent=intent, detected_topic=detected_topic or "this topic")
        elif confidence >= confidence_config["low_confidence"]["threshold"]:
            template = confidence_config["low_confidence"]["response"]
            suggestions = self.boundaries["out_of_scope"]["suggest_alternatives"]
            return template.format(suggested_question=suggestions[0])
        else:
            template = confidence_config["no_confidence"]["response"]
            examples = self.boundaries["out_of_scope"]["suggest_alternatives"]
            return template.format(example_questions=", ".join(examples[:2]))
    
    def get_consent_mapping(self, consent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get consent mapping for a specific consent type.
        
        Args:
            consent_type: Type of consent (e.g., "affirmative_consent")
            
        Returns:
            Consent mapping dictionary or None
        """
        consent_mappings = self.mappings.get("consent_mappings", {})
        return consent_mappings.get("consent_types", {}).get(consent_type)
    
    def get_regulation_mapping(self, regulation_key: str) -> Optional[Dict[str, Any]]:
        """
        Get regulation mapping for a specific regulation.
        
        Args:
            regulation_key: Regulation identifier (e.g., "gdpr", "gabon")
            
        Returns:
            Regulation mapping dictionary or None
        """
        regulation_mappings = self.mappings.get("regulation_mappings", {})
        return regulation_mappings.get("regulations", {}).get(regulation_key.lower())
    
    def get_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all intent patterns for MCP detection.
        
        Returns:
            Dictionary of intent patterns
        """
        return self.mappings.get("intent_patterns", {}).get("intent_patterns", {})
    
    def get_mcp_priority_tools(self) -> List[str]:
        """
        Get priority MCP tools from profile.
        
        Returns:
            List of priority tool names
        """
        return self.profile.get("mcp_configuration", {}).get("priority_tools", [])
    
    def is_tool_disabled(self, tool_name: str) -> bool:
        """
        Check if a tool is disabled in current profile.
        
        Args:
            tool_name: Name of the MCP tool
            
        Returns:
            True if tool is disabled
        """
        disabled_tools = self.profile.get("mcp_configuration", {}).get("disabled_tools", [])
        return tool_name in disabled_tools
    
    def get_tool_restrictions(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get restrictions for a specific tool.
        
        Args:
            tool_name: Name of the MCP tool
            
        Returns:
            Tool restrictions dictionary or None
        """
        restrictions = self.profile.get("mcp_configuration", {}).get("tool_restrictions", {})
        return restrictions.get(tool_name)
    
    def validate_chunk_count(self, chunk_count: int) -> Tuple[bool, Optional[str]]:
        """
        Validate if chunk count is within expected range.
        
        Args:
            chunk_count: Number of chunks retrieved
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        validation = self.filters["validation"]
        
        if chunk_count < validation["min_results"]:
            return False, f"Warning: Only {chunk_count} chunks found, expected at least {validation['min_results']}"
        elif chunk_count > validation["max_results"]:
            return False, f"Warning: {chunk_count} chunks found, expected at most {validation['max_results']}"
        
        return True, None
    
    def get_expected_jurisdictions(self) -> List[str]:
        """
        Get list of expected jurisdictions from profile.
        
        Returns:
            List of jurisdiction names
        """
        return self.profile.get("jurisdictions", {}).get("expected", [])
    
    def reload_configurations(self):
        """Reload all configurations from disk"""
        logger.info("Reloading configurations...")
        self._load_configurations()
        # Clear any caches
        self.get_search_filters.cache_clear()
        logger.info("Configurations reloaded successfully")


# Singleton instance
_config_service: Optional[ConfigService] = None


def get_config_service() -> ConfigService:
    """
    Get the singleton ConfigService instance.
    
    Returns:
        ConfigService instance
    """
    global _config_service
    if _config_service is None:
        _config_service = ConfigService()
    return _config_service