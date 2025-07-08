"""
Dynamic TOP-K strategy for different query types and stages
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TopKConfig:
    """Configuration for TOP-K values at different stages"""
    initial_k: int  # Query Manager recommendation
    retrieval_k: int  # Retriever fetch size
    context_k: int  # Response Generator context size
    min_k: int = 5
    max_k: int = 200


class TopKStrategy:
    """
    Determines optimal TOP-K values based on query characteristics
    """
    
    def __init__(self):
        # Intent-based initial recommendations
        self.intent_configs = {
            # Focused searches
            "definition_lookup": TopKConfig(10, 20, 5),
            "specific_regulation": TopKConfig(20, 30, 10),
            
            # Moderate searches
            "general_query": TopKConfig(30, 50, 15),
            "compliance_check": TopKConfig(30, 50, 15),
            "implementation_guide": TopKConfig(40, 60, 20),
            
            # Broad searches
            "compare_regulations": TopKConfig(50, 80, 25),
            "retrieve_and_summarize": TopKConfig(50, 100, 30),
            "multi_source_analysis": TopKConfig(80, 150, 40),
            
            # Comprehensive searches
            "gap_analysis": TopKConfig(60, 100, 30),
            "risk_assessment": TopKConfig(50, 80, 25),
            
            # Default
            "default": TopKConfig(30, 50, 20)
        }
        
        # Scope multipliers
        self.scope_multipliers = {
            "specific": 1.0,
            "comparative": 1.5,
            "comprehensive": 2.0
        }
        
        # Regulation count multipliers
        self.regulation_multipliers = {
            1: 1.0,      # Single regulation
            2: 1.3,      # Two regulations
            3: 1.5,      # Three regulations
            "ALL": 2.0   # All regulations
        }
    
    def get_initial_topk(
        self, 
        intent: str, 
        scope: str,
        regulations: list,
        confidence: float
    ) -> int:
        """
        Get initial TOP-K recommendation for Query Manager
        """
        # Get base config
        config = self.intent_configs.get(intent, self.intent_configs["default"])
        base_k = config.initial_k
        
        # Apply scope multiplier
        scope_mult = self.scope_multipliers.get(scope, 1.0)
        
        # Apply regulation multiplier
        if "ALL" in regulations:
            reg_mult = self.regulation_multipliers["ALL"]
        else:
            reg_count = len(regulations) if regulations else 1
            reg_mult = self.regulation_multipliers.get(reg_count, 1.2)
        
        # Apply confidence adjustment (lower confidence = cast wider net)
        confidence_mult = 1.0 + (1.0 - confidence) * 0.5
        
        # Calculate final K
        final_k = int(base_k * scope_mult * reg_mult * confidence_mult)
        
        # Clamp to bounds
        final_k = max(config.min_k, min(final_k, config.max_k))
        
        logger.info(f"Initial TOP-K for {intent}/{scope}: {final_k} (base: {base_k})")
        return final_k
    
    def adjust_retrieval_topk(
        self,
        initial_k: int,
        query_complexity: str,
        has_metadata_filters: bool,
        previous_search_success: Optional[bool] = None
    ) -> int:
        """
        Adjust TOP-K for retrieval stage based on conditions
        """
        retrieval_k = initial_k
        
        # If we have good metadata filters, we can fetch more
        if has_metadata_filters:
            retrieval_k = int(retrieval_k * 1.5)
        
        # Complex queries need more results
        complexity_multipliers = {
            "simple": 1.0,
            "moderate": 1.2,
            "complex": 1.5
        }
        retrieval_k = int(retrieval_k * complexity_multipliers.get(query_complexity, 1.2))
        
        # If previous search failed, increase K
        if previous_search_success is False:
            retrieval_k = int(retrieval_k * 1.5)
        
        # Clamp to bounds
        retrieval_k = max(10, min(retrieval_k, 200))
        
        logger.info(f"Adjusted retrieval TOP-K: {retrieval_k}")
        return retrieval_k
    
    def get_context_topk(
        self,
        retrieval_k: int,
        intent: str,
        output_format: str,
        token_budget: int = 8000
    ) -> int:
        """
        Determine how many results to include in context for response generation
        """
        # Get base config
        config = self.intent_configs.get(intent, self.intent_configs["default"])
        base_context_k = config.context_k
        
        # Adjust based on output format
        format_multipliers = {
            "summary": 0.8,      # Fewer needed for summary
            "detailed": 1.2,     # More for detailed response
            "comparison": 1.5,   # More for comparisons
            "table": 1.0,        # Standard for tables
            "list": 0.9          # Slightly fewer for lists
        }
        
        context_k = int(base_context_k * format_multipliers.get(output_format, 1.0))
        
        # Estimate tokens per result (rough estimate)
        avg_tokens_per_result = 300
        max_results_by_tokens = token_budget // avg_tokens_per_result
        
        # Take minimum of calculated K and token budget limit
        context_k = min(context_k, max_results_by_tokens, retrieval_k)
        
        logger.info(f"Context TOP-K for response generation: {context_k}")
        return context_k
    
    def get_progressive_topk_strategy(
        self,
        intent: str,
        scope: str
    ) -> Dict[str, int]:
        """
        Get a progressive search strategy with multiple rounds
        """
        strategies = {
            "comprehensive": {
                "round_1": 200,
                "round_2": 300,
                "round_3": 400,
                "max_rounds": 3
            },
            "targeted": {
                "round_1": 50,
                "round_2": 100,
                "round_3": 150,
                "max_rounds": 3
            },
            "exploratory": {
                "round_1": 100,
                "round_2": 200,
                "round_3": 300,
                "max_rounds": 3
            }
        }
        
        # Determine strategy type
        if scope == "comprehensive" or intent == "multi_source_analysis":
            return strategies["comprehensive"]
        elif scope == "specific" or intent in ["definition_lookup", "specific_regulation"]:
            return strategies["targeted"]
        else:
            return strategies["exploratory"]
    
    def should_expand_search(
        self,
        current_results_count: int,
        min_confidence_score: float,
        current_k: int,
        max_k: int = 200
    ) -> Tuple[bool, int]:
        """
        Determine if search should be expanded with higher K
        """
        # Don't expand if already at max
        if current_k >= max_k:
            return False, current_k
        
        # Expand if too few results
        if current_results_count < 5:
            new_k = min(current_k * 2, max_k)
            return True, new_k
        
        # Expand if confidence is too low
        if min_confidence_score < 0.7:
            new_k = min(int(current_k * 1.5), max_k)
            return True, new_k
        
        return False, current_k