"""
Context Builder - Optimizes prompt construction from search results
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re
import tiktoken
from src.models.search import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextSegment:
    """Represents a segment of context with metadata"""
    content: str
    source: str
    clause_id: str
    relevance_score: float
    metadata: Dict[str, Any]
    token_count: int
    
    # Enhanced citation fields
    regulation: Optional[str] = None
    article_number: Optional[str] = None
    section_number: Optional[str] = None
    subsection: Optional[str] = None
    clause_title: Optional[str] = None
    clause_number: Optional[str] = None
    page_number: Optional[int] = None
    hierarchy_path: Optional[str] = None
    effective_date: Optional[str] = None
    
    def to_citation(self, style: str = "standard") -> str:
        """Generate citation in requested format"""
        if self.regulation and (self.article_number or self.section_number):
            if style == "legal":
                parts = [self.regulation]
                if self.article_number:
                    parts.append(self.article_number)
                elif self.section_number:
                    parts.append(f"Section {self.section_number}")
                if self.subsection:
                    parts.append(self.subsection)
                return f"[{' '.join(parts)}]"
            elif style == "detailed" and self.page_number:
                base = self.to_citation("legal")[1:-1]  # Remove brackets
                return f"[{base}, p. {self.page_number}]"
        # Fallback to current format
        return f"[Doc: {self.source}, Clause: {self.clause_id}]"


@dataclass
class BuiltContext:
    """Result of context building process"""
    segments: List[ContextSegment]
    total_tokens: int
    primary_topics: List[str]
    query_intent: str
    metadata_summary: Dict[str, Any]
    
    def get_formatted_context(self) -> str:
        """Get formatted context string for prompt"""
        context_parts = []
        
        # Group by source document
        by_source = {}
        for segment in self.segments:
            if segment.source not in by_source:
                by_source[segment.source] = []
            by_source[segment.source].append(segment)
        
        # Format each source
        for source, segments in by_source.items():
            context_parts.append(f"=== {source} ===")
            
            for segment in segments:
                # Build citation reference
                citation_ref = segment.to_citation("legal")
                
                # Add enhanced metadata context
                metadata_str = ""
                if segment.regulation:
                    metadata_str += f"Regulation: {segment.regulation}\n"
                if segment.article_number:
                    metadata_str += f"Article: {segment.article_number}\n"
                if segment.section_number:
                    metadata_str += f"Section: {segment.section_number}\n"
                if segment.clause_title:
                    metadata_str += f"Title: {segment.clause_title}\n"
                if segment.metadata.get("effective_date") or segment.effective_date:
                    metadata_str += f"Effective Date: {segment.effective_date or segment.metadata.get('effective_date')}\n"
                if segment.metadata.get("topics"):
                    metadata_str += f"Topics: {', '.join(segment.metadata['topics'])}\n"
                
                if metadata_str:
                    context_parts.append(metadata_str.strip())
                
                # Add the actual content with proper citation, wrapped in triple backticks
                context_parts.append(f"\n{citation_ref} - Clause {segment.clause_id}:")
                context_parts.append("```")
                context_parts.append(segment.content)
                context_parts.append("```")
                context_parts.append("")  # Empty line for readability
        
        formatted_context = "\n".join(context_parts)
        
        # Debug log for Article 13 queries
        if "article 13" in formatted_context.lower():
            logger.info(f"Formatted context contains Article 13. First 500 chars:\n{formatted_context[:500]}...")
            
        return formatted_context


class ContextBuilder:
    """Builds optimized context from search results for LLM prompts"""
    
    def __init__(self, max_context_tokens: int = 8000, max_chunks_for_complex: int = 12):
        self.max_context_tokens = max_context_tokens
        self.max_chunks_for_complex = max_chunks_for_complex  # Limit chunks for complex queries
        
        # Initialize tiktoken for accurate token counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Token estimation fallback
        self.chars_per_token = 4
        
        # Context optimization strategies
        self.deduplication_threshold = 0.85  # Similarity threshold for deduplication
        self.min_relevance_score = 0.01  # Lowered to include semantic search results
    
    async def build_context(
        self,
        search_results: List[SearchResult],
        query_intent: str,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        is_multi_jurisdiction: bool = False,
        mentioned_jurisdictions: List[str] = None
    ) -> BuiltContext:
        """Build optimized context from search results"""
        
        # Extract and prioritize segments
        segments = await self._extract_segments(search_results, query_intent)
        
        # Deduplicate similar content
        segments = self._deduplicate_segments(segments)
        
        # For complex/multi-jurisdiction queries, limit chunks before token optimization
        if is_multi_jurisdiction and len(segments) > self.max_chunks_for_complex:
            logger.info(f"Multi-jurisdiction query: limiting from {len(segments)} to {self.max_chunks_for_complex} chunks")
            # Ensure we get chunks from each mentioned jurisdiction
            segments = self._balance_jurisdictions(segments, mentioned_jurisdictions, self.max_chunks_for_complex)
        
        # Optimize for token limit
        segments = self._optimize_for_tokens(segments, user_query)
        
        # Extract metadata summary
        metadata_summary = self._summarize_metadata(segments)
        
        # Identify primary topics
        primary_topics = self._identify_primary_topics(segments)
        
        # Calculate total tokens
        total_tokens = sum(s.token_count for s in segments)
        
        return BuiltContext(
            segments=segments,
            total_tokens=total_tokens,
            primary_topics=primary_topics,
            query_intent=query_intent,
            metadata_summary=metadata_summary
        )
    
    async def _extract_segments(
        self,
        search_results: List[SearchResult],
        query_intent: str
    ) -> List[ContextSegment]:
        """Extract context segments from search results"""
        
        segments = []
        
        for result in search_results:
            # Skip low relevance results
            if result.score < self.min_relevance_score:
                continue
            
            # Extract citation metadata
            metadata = result.chunk.metadata
            if not isinstance(metadata, dict):
                logger.warning(f"chunk.metadata is not a dict, it's a {type(metadata)}")
                metadata = {}  # Default to empty dict
            
            # Parse article/section numbers from metadata or content
            article_number = self._extract_article_number(metadata, result.chunk.content)
            section_number = self._extract_section_number(metadata, result.chunk.content)
            
            # Enhanced citation fields
            # Use generated_document_name instead of unreliable regulation field
            inferred_reg = self._infer_regulation(result.chunk.document_name)
            generated_doc_name = metadata.get("generated_document_name", "").strip('"')
            regulation = inferred_reg or generated_doc_name
            
            # Debug logging for Article 13 queries
            if "article 13" in result.chunk.content.lower():
                logger.info(f"Article 13 found in chunk: doc={result.chunk.document_name}, "
                           f"inferred_reg={inferred_reg}, generated_doc_name={generated_doc_name}, "
                           f"final_regulation={regulation}")
            
            # Create segment with enhanced citation fields
            segment = ContextSegment(
                content=result.chunk.content,
                source=result.chunk.document_name,
                clause_id=metadata.get("clause_id", result.chunk.id),
                relevance_score=result.score,
                metadata=metadata,
                token_count=self._estimate_tokens(result.chunk.content),
                regulation=regulation,
                article_number=article_number,
                section_number=section_number,
                subsection=metadata.get("subsection"),
                clause_title=metadata.get("clause_title"),
                clause_number=metadata.get("clause_number"),
                page_number=result.chunk.page_number,
                hierarchy_path=metadata.get("hierarchy_path"),
                effective_date=metadata.get("effective_date")
            )
            
            # Boost relevance for intent-matching content
            if query_intent == "comparison" and "compare" in result.chunk.content.lower():
                segment.relevance_score *= 1.2
            elif query_intent == "specific_requirement" and any(
                keyword in result.chunk.content.lower() 
                for keyword in ["must", "shall", "require", "mandate"]
            ):
                segment.relevance_score *= 1.1
            
            segments.append(segment)
        
        # Sort by relevance
        segments.sort(key=lambda s: s.relevance_score, reverse=True)
        
        return segments
    
    def _deduplicate_segments(
        self,
        segments: List[ContextSegment]
    ) -> List[ContextSegment]:
        """Remove duplicate or highly similar segments"""
        
        unique_segments = []
        
        for segment in segments:
            is_duplicate = False
            
            for unique in unique_segments:
                # Simple similarity check (can be enhanced with better algorithms)
                similarity = self._calculate_similarity(
                    segment.content,
                    unique.content
                )
                
                if similarity > self.deduplication_threshold:
                    # Keep the one with higher relevance
                    if segment.relevance_score > unique.relevance_score:
                        unique_segments.remove(unique)
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_segments.append(segment)
        
        return unique_segments
    
    def _balance_jurisdictions(
        self,
        segments: List[ContextSegment],
        mentioned_jurisdictions: List[str],
        max_chunks: int
    ) -> List[ContextSegment]:
        """Balance segments across mentioned jurisdictions"""
        
        if not mentioned_jurisdictions:
            # No specific jurisdictions, just take top chunks by score
            return sorted(segments, key=lambda s: s.relevance_score, reverse=True)[:max_chunks]
        
        # Group segments by jurisdiction
        by_jurisdiction = {j: [] for j in mentioned_jurisdictions}
        other_segments = []
        
        for segment in segments:
            jurisdiction = segment.metadata.get('jurisdiction', '').lower()
            matched = False
            
            for mentioned in mentioned_jurisdictions:
                if mentioned.lower() in jurisdiction:
                    by_jurisdiction[mentioned].append(segment)
                    matched = True
                    break
            
            if not matched:
                other_segments.append(segment)
        
        # Calculate chunks per jurisdiction
        chunks_per_jurisdiction = max_chunks // len(mentioned_jurisdictions)
        remaining_slots = max_chunks % len(mentioned_jurisdictions)
        
        balanced_segments = []
        
        # Add top chunks from each jurisdiction
        for jurisdiction, jurisdiction_segments in by_jurisdiction.items():
            # Sort by relevance within jurisdiction
            sorted_segments = sorted(jurisdiction_segments, key=lambda s: s.relevance_score, reverse=True)
            
            # Take allocated chunks plus one extra if we have remaining slots
            take_count = chunks_per_jurisdiction
            if remaining_slots > 0:
                take_count += 1
                remaining_slots -= 1
            
            balanced_segments.extend(sorted_segments[:take_count])
            
            logger.info(f"Added {min(take_count, len(sorted_segments))} chunks from {jurisdiction}")
        
        # Fill any remaining slots with highest scoring other segments
        if len(balanced_segments) < max_chunks and other_segments:
            remaining = max_chunks - len(balanced_segments)
            sorted_others = sorted(other_segments, key=lambda s: s.relevance_score, reverse=True)
            balanced_segments.extend(sorted_others[:remaining])
        
        return balanced_segments
    
    def _optimize_for_tokens(
        self,
        segments: List[ContextSegment],
        user_query: str
    ) -> List[ContextSegment]:
        """Optimize segments to fit within token limit"""
        
        # Reserve tokens for query and response
        query_tokens = self._estimate_tokens(user_query)
        # Reserve tokens for response generation and system prompt (reduced from 2500)
        reserved_tokens = query_tokens + 1800  # Reduced to allow more context
        available_tokens = max(1000, self.max_context_tokens - reserved_tokens)  # Ensure min context
        
        logger.info(f"Token budget: query={query_tokens}, reserved={reserved_tokens}, available={available_tokens}")
        
        optimized = []
        current_tokens = 0
        
        for i, segment in enumerate(segments):
            if current_tokens + segment.token_count <= available_tokens:
                optimized.append(segment)
                current_tokens += segment.token_count
            else:
                # Try to truncate the segment
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Only include if meaningful
                    truncated_content = self._truncate_to_tokens(
                        segment.content,
                        remaining_tokens - 50  # Leave some buffer
                    )
                    segment.content = truncated_content + "..."
                    segment.token_count = self._estimate_tokens(segment.content)
                    optimized.append(segment)
                    current_tokens += segment.token_count
                    
                logger.info(f"Truncated context at segment {i+1}/{len(segments)}, total tokens: {current_tokens}")
                break
        
        return optimized
    
    def _summarize_metadata(
        self,
        segments: List[ContextSegment]
    ) -> Dict[str, Any]:
        """Summarize metadata across all segments"""
        
        summary = {
            "regulatory_bodies": set(),
            "topics": {},
            "date_range": {"earliest": None, "latest": None},
            "jurisdictions": set(),
            "document_types": set()
        }
        
        for segment in segments:
            # Regulatory bodies
            if body := segment.metadata.get("regulatory_body"):
                summary["regulatory_bodies"].add(body)
            
            # Topics with frequency
            if topics := segment.metadata.get("topics"):
                for topic in topics:
                    summary["topics"][topic] = summary["topics"].get(topic, 0) + 1
            
            # Date range
            if date_str := segment.metadata.get("effective_date"):
                try:
                    date = datetime.fromisoformat(date_str)
                    if not summary["date_range"]["earliest"] or date < summary["date_range"]["earliest"]:
                        summary["date_range"]["earliest"] = date
                    if not summary["date_range"]["latest"] or date > summary["date_range"]["latest"]:
                        summary["date_range"]["latest"] = date
                except:
                    pass
            
            # Jurisdictions
            if jurisdiction := segment.metadata.get("jurisdiction"):
                summary["jurisdictions"].add(jurisdiction)
            
            # Document types
            if doc_type := segment.metadata.get("document_type"):
                summary["document_types"].add(doc_type)
        
        # Convert sets to lists for JSON serialization
        summary["regulatory_bodies"] = list(summary["regulatory_bodies"])
        summary["jurisdictions"] = list(summary["jurisdictions"])
        summary["document_types"] = list(summary["document_types"])
        
        # Sort topics by frequency
        summary["topics"] = dict(
            sorted(summary["topics"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return summary
    
    def _identify_primary_topics(
        self,
        segments: List[ContextSegment]
    ) -> List[str]:
        """Identify the primary topics from segments"""
        
        topic_scores = {}
        
        for segment in segments:
            if topics := segment.metadata.get("topics"):
                for topic in topics:
                    # Weight by relevance score
                    topic_scores[topic] = topic_scores.get(topic, 0) + segment.relevance_score
        
        # Get top 5 topics
        sorted_topics = sorted(
            topic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [topic for topic, _ in sorted_topics[:5]]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to simple estimation
            return len(text) // self.chars_per_token
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max tokens"""
        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate to exact token count
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            # Try to end at a sentence boundary
            last_period = truncated_text.rfind('.')
            last_newline = truncated_text.rfind('\n')
            
            cut_point = max(last_period, last_newline)
            if cut_point > len(truncated_text) * 0.8:  # If we're not losing too much
                return truncated_text[:cut_point + 1]
            
            return truncated_text
        except Exception:
            # Fallback to character-based truncation
            max_chars = max_tokens * self.chars_per_token
            
            if len(text) <= max_chars:
                return text
            
            truncated = text[:max_chars]
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            
            cut_point = max(last_period, last_newline)
            if cut_point > max_chars * 0.8:
                return truncated[:cut_point + 1]
            
            return truncated
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple implementation)"""
        # Normalize texts
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Simple word overlap calculation
        words1 = set(re.findall(r'\w+', text1_lower))
        words2 = set(re.findall(r'\w+', text2_lower))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_article_number(self, metadata: Dict[str, Any], content: str) -> Optional[str]:
        """Extract article number from metadata or content"""
        # First try metadata
        if metadata.get("article_number"):
            return metadata["article_number"]
        
        # Try to extract from content using enhanced regex patterns
        article_patterns = [
            # GDPR patterns with subsections
            r'Article\s+(\d+)\s*\((\d+)\)',  # Article 7(3)
            r'Article\s+(\d+)\((\d+)\)\((\w+)\)',  # Article 13(1)(a)
            r'Article\s+(\d+)\s+paragraph\s+(\d+)',  # Article 7 paragraph 3
            r'Article\s+(\d+(?:\.\d+)?)',  # Article 7 or Article 7.3
            r'Art\.\s*(\d+(?:\.\d+)?)',  # Art. 7
            r'ARTICLE\s+([IVXLCDM]+)',  # Roman numerals
        ]
        
        # Check more content for better extraction
        for pattern in article_patterns:
            match = re.search(pattern, content[:1000], re.IGNORECASE)  # Check first 1000 chars
            if match:
                if len(match.groups()) > 1:
                    # Has subsection
                    article = match.group(1)
                    subsection = match.group(2)
                    if len(match.groups()) > 2:
                        # Has sub-subsection
                        return f"Article {article}({subsection})({match.group(3)})"
                    return f"Article {article}({subsection})"
                return f"Article {match.group(1)}"
        
        return None
    
    def _extract_section_number(self, metadata: Dict[str, Any], content: str) -> Optional[str]:
        """Extract section number from metadata or content"""
        # First try metadata
        if metadata.get("section_number"):
            return metadata["section_number"]
        
        # Try to extract from content using enhanced regex patterns
        section_patterns = [
            # CCPA patterns
            r'Section\s+(\d{4}\.\d+)\s*\(([a-z])\)\s*\((\d+)\)',  # Section 1798.100(a)(1)
            r'§\s*(\d{4}\.\d+)\s*\(([a-z])\)',  # § 1798.100(a)
            r'Section\s+(\d{4}\.\d+)',  # Section 1798.100
            # HIPAA patterns  
            r'§\s*(\d{3}\.\d+)\s*\(([a-z])\)\s*\((\d+)\)',  # § 164.502(a)(1)
            r'Section\s+(\d{3}\.\d+)\s*\(([a-z])\)',  # Section 164.502(a)
            # FERPA patterns
            r'§\s*(\d{2}\.\d+)\s*\(([a-z])\)\s*\((\d+)\)',  # § 99.31(a)(1)
            # General patterns
            r'Section\s+(\d+(?:\.\d+)?(?:\([a-z]+\))?)',
            r'§\s*(\d+(?:\.\d+)?(?:\([a-z]+\))?)',
            r'Sec\.\s*(\d+(?:\.\d+)?)',
        ]
        
        # Check more content for better extraction
        for pattern in section_patterns:
            match = re.search(pattern, content[:1000], re.IGNORECASE)
            if match:
                # Build the full section reference
                groups = match.groups()
                if len(groups) >= 3:
                    # Has subsection and sub-subsection
                    return f"{groups[0]}({groups[1]})({groups[2]})"
                elif len(groups) >= 2:
                    # Has subsection
                    return f"{groups[0]}({groups[1]})"
                return match.group(1)
        
        return None
    
    def _infer_regulation(self, document_name: str) -> Optional[str]:
        """Infer regulation from document name or content patterns"""
        doc_upper = document_name.upper()
        
        # Check document name patterns
        if 'GABON' in doc_upper or 'ORDINANCE' in doc_upper and 'PERSONAL DATA' in doc_upper:
            return 'Personal Data Protection Ordinance - Gabon'
        elif 'GDPR' in doc_upper or 'GENERAL DATA PROTECTION' in doc_upper:
            return 'GDPR'
        elif 'CCPA' in doc_upper or 'CALIFORNIA CONSUMER PRIVACY' in doc_upper:
            return 'CCPA'
        elif 'HIPAA' in doc_upper or 'HEALTH INSURANCE PORTABILITY' in doc_upper:
            return 'HIPAA'
        elif 'FERPA' in doc_upper or 'FAMILY EDUCATIONAL RIGHTS' in doc_upper:
            return 'FERPA'
        elif 'REGULATION' in doc_upper and '2016/679' in doc_upper:
            return 'GDPR'
        elif '1798' in doc_upper:  # California Civil Code
            return 'CCPA'
        elif '164' in doc_upper and ('SECURITY' in doc_upper or 'PRIVACY' in doc_upper):
            return 'HIPAA'
        elif '99.31' in doc_upper or '1232G' in doc_upper:
            return 'FERPA'
        elif 'DENMARK' in doc_upper and 'DATA PROTECTION' in doc_upper:
            return 'Danish Data Protection Act'
        elif 'ESTONIA' in doc_upper and 'PERSONAL DATA' in doc_upper:
            return 'Estonian Personal Data Protection Act'
        elif 'COSTA RICA' in doc_upper:
            return 'Costa Rica Data Protection Law'
        
        return None
    
    def create_comparison_context(
        self,
        segments_by_regulation: Dict[str, List[ContextSegment]]
    ) -> str:
        """Create specialized context for comparison queries"""
        
        comparison_parts = []
        
        for regulation, segments in segments_by_regulation.items():
            comparison_parts.append(f"\n=== {regulation} Requirements ===")
            
            for segment in segments[:3]:  # Limit to top 3 per regulation
                comparison_parts.append(f"\nClause {segment.clause_id}:")
                comparison_parts.append(segment.content)
                
                # Add key metadata
                if segment.metadata.get("effective_date"):
                    comparison_parts.append(
                        f"Effective: {segment.metadata['effective_date']}"
                    )
        
        return "\n".join(comparison_parts)
    
    def create_timeline_context(
        self,
        segments: List[ContextSegment]
    ) -> str:
        """Create context organized by timeline"""
        
        # Sort by effective date
        dated_segments = [
            s for s in segments 
            if s.metadata.get("effective_date")
        ]
        
        dated_segments.sort(
            key=lambda s: s.metadata.get("effective_date", "")
        )
        
        timeline_parts = ["\n=== Regulatory Timeline ==="]
        
        for segment in dated_segments:
            date = segment.metadata.get("effective_date", "Unknown")
            timeline_parts.append(f"\n{date} - {segment.source}")
            timeline_parts.append(f"Clause {segment.clause_id}: {segment.content[:200]}...")
        
        return "\n".join(timeline_parts)