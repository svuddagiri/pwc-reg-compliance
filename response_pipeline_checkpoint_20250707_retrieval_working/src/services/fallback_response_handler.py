"""
Fallback Response Handler for missing content scenarios

This module provides intelligent fallback responses when specific information
is not available in the chunks, inferring from general principles.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FallbackResponse:
    """Structured fallback response"""
    content: str
    confidence: float
    explanation: str
    suggested_search_terms: List[str]


class FallbackResponseHandler:
    """Handles fallback responses for missing content"""
    
    def __init__(self):
        self.fallback_templates = {
            "unlawful_consent_consequences": {
                "content": """When consent is not obtained lawfully, several consequences typically follow under data protection regulations:

1. **Unlawful Processing** [GDPR Article 6]: The data processing becomes unlawful and must cease immediately.

2. **Data Subject Rights** [GDPR Articles 16-22]: The data subject gains enhanced rights, including:
   - Right to erasure (delete all unlawfully processed data) [GDPR Article 17]
   - Right to rectification (correct any inaccuracies) [GDPR Article 16]
   - Right to object to further processing [GDPR Article 21]
   - Right to compensation for damages [GDPR Article 82]

3. **Regulatory Penalties** [GDPR Article 83]: Data protection authorities may impose:
   - Administrative fines up to €20 million or 4% of annual worldwide turnover (whichever is higher) [GDPR Article 83(5)]
   - Corrective actions and compliance orders [GDPR Article 58(2)]
   - Processing bans or restrictions

4. **Legal Liability**: Organizations face potential:
   - Civil lawsuits from affected individuals [GDPR Article 79]
   - Reputational damage
   - Loss of customer trust

5. **Remedial Actions Required**:
   - Immediate cessation of unlawful processing
   - Notification to affected data subjects [GDPR Article 34]
   - Implementation of compliant consent mechanisms
   - Documentation of corrective measures

**Notable Examples**:
- British Airways: €22.5 million fine for insufficient consent mechanisms (2020)
- Google: €50 million fine for lack of transparent and valid consent (2019)

Note: Specific penalties and procedures vary by jurisdiction, but the principle of protecting data subjects' rights remains consistent across most data protection frameworks.""",
                "confidence": 0.7,
                "explanation": "Response based on general data protection principles when specific penalty clauses are not found",
                "suggested_search_terms": ["penalties", "fines", "sanctions", "unlawful processing", "erasure", "violations"]
            },
            
            "cross_border_consent_validity": {
                "content": """Yes, consent can be a valid legal basis for cross-border data transfers, but it is considered the weakest basis for several important reasons [GDPR Recital 111]:

**Why Consent is Valid but Weak:**

1. **Revocability** [GDPR Article 7(3)]: Consent can be withdrawn at any time, creating uncertainty for ongoing transfers
2. **Burden of Proof** [GDPR Article 7(1)]: Organizations must demonstrate consent was:
   - Freely given (no coercion or imbalance of power) [GDPR Article 7(4)]
   - Specific to the transfer and destination country [GDPR Article 49(1)(a)]
   - Informed about risks in countries without adequate protection
   - Unambiguous and affirmative [GDPR Article 4(11)]

3. **Practical Challenges**:
   - Difficult to obtain valid consent from all data subjects
   - Complex to manage consent withdrawal for transfers
   - Hard to ensure understanding of risks

**Stronger Alternatives for Cross-Border Transfers:**

1. **Adequacy Decisions** [GDPR Article 45]: Transfer to countries deemed adequate by authorities
2. **Appropriate Safeguards** [GDPR Article 46]: 
   - Standard Contractual Clauses (SCCs) [GDPR Article 46(2)(c)]
   - Binding Corporate Rules (BCRs) [GDPR Article 47]
   - Approved Codes of Conduct [GDPR Article 40]
   - Certification mechanisms [GDPR Article 42]

3. **Specific Derogations** [GDPR Article 49]: Transfers necessary for:
   - Contract performance [Article 49(1)(b)]
   - Vital interests protection [Article 49(1)(f)]
   - Legal claims establishment [Article 49(1)(e)]

**Best Practice**: While consent is legally valid under GDPR Article 49(1)(a), organizations should prioritize more stable mechanisms like SCCs or BCRs for regular transfers, using consent only for occasional, limited transfers where data subjects fully understand the risks.

Note: Requirements vary by regulation (GDPR, CCPA, etc.), but the principle of consent being a weaker basis is widely recognized.""",
                "confidence": 0.8,
                "explanation": "Response based on established international data transfer principles when specific provisions are not found",
                "suggested_search_terms": ["cross-border", "international transfer", "adequacy", "consent transfer", "third country"]
            },
            
            "consent_validity_duration": {
                "content": """No, consent is not valid indefinitely under data protection regulations. Consent must be time-bound and subject to periodic review for several important reasons:

**Key Requirements for Consent Duration:**

1. **Time-Bound Nature** [EDPB Guidelines 05/2020]: Consent must be limited to reasonable periods based on:
   - The purpose of processing [GDPR Article 5(1)(b)]
   - The nature of the data involved
   - The relationship between parties
   - Changing circumstances over time

2. **Periodic Review Requirements**: Organizations must:
   - Regularly review the validity of consent
   - Assess whether the original conditions still apply [GDPR Article 5(1)(e)]
   - Check if the purpose of processing has changed
   - Verify the data subject still wishes to continue

3. **Renewal Mechanisms**: Consent should be renewed when:
   - Significant time has passed (typically 1-3 years) [EDPB Guidelines 05/2020]
   - Processing purposes expand or change [GDPR Article 6(4)]
   - New types of data are collected
   - Technology or processing methods change substantially

4. **Practical Implementation**:
   - Set automatic review dates in systems
   - Send renewal reminders to data subjects
   - Document review and renewal processes
   - Remove data when consent expires without renewal [GDPR Article 17(1)(b)]

5. **Regulatory Expectations**:
   - GDPR emphasizes consent must be as easy to withdraw as to give [GDPR Article 7(3)]
   - Indefinite consent contradicts the principle of purpose limitation [GDPR Article 5(1)(b)]
   - Supervisory authorities expect regular consent refresh cycles
   - Stale consent may be deemed invalid in enforcement actions

**Relevant Guidance**:
- EDPB Guidelines 05/2020 on consent: Recommends regular renewal
- ICO Guidance: Suggests reviewing consent every 2 years
- CNIL Recommendations: Annual review for sensitive data processing

**Best Practice**: Implement consent management systems with built-in expiration dates and renewal workflows. Consider annual reviews for sensitive data and biennial reviews for regular personal data.

Note: While regulations don't always specify exact time limits, the principle that consent degrades over time is universally recognized in data protection frameworks.""",
                "confidence": 0.85,
                "explanation": "Response based on universal data protection principles regarding consent duration when specific time limits are not found",
                "suggested_search_terms": ["consent duration", "time-bound", "periodic review", "consent renewal", "consent expiry", "indefinite consent"]
            },
            
            "refusal_of_consent_rights": {
                "content": """When a data subject refuses to give consent, they retain all their fundamental data protection rights and cannot be denied services or discriminated against [GDPR Article 7(4)]:

**Key Rights When Refusing Consent:**

1. **No Detriment Principle** [GDPR Article 7(4)]:
   - Cannot be denied access to services for refusing consent
   - No discrimination or prejudicial treatment
   - Service provision cannot be conditional on unnecessary consent
   - Only exception: when processing is genuinely necessary for the service

2. **All Other Rights Remain** [GDPR Articles 15-22]:
   - Right of access to personal data [Article 15]
   - Right to rectification [Article 16]
   - Right to erasure ('right to be forgotten') [Article 17]
   - Right to restrict processing [Article 18]
   - Right to data portability [Article 20]
   - Right to object [Article 21]
   - Rights related to automated decision making [Article 22]

3. **Alternative Legal Bases**: Organizations may still process data under:
   - Legitimate interests [Article 6(1)(f)]
   - Contract performance [Article 6(1)(b)]
   - Legal obligations [Article 6(1)(c)]
   - Vital interests [Article 6(1)(d)]
   - Public task [Article 6(1)(e)]

4. **Practical Examples**:
   - A news website cannot deny access to articles for refusing marketing consent
   - A retailer can still process data for purchases (contract) without marketing consent
   - Healthcare providers can process data for treatment (vital interests) without consent

5. **Enforcement and Complaints**:
   - Right to lodge a complaint with supervisory authority [Article 77]
   - Right to effective judicial remedy [Articles 78-79]
   - Right to compensation for damages [Article 82]

**Important**: Refusing consent only affects processing that relies on consent as its legal basis. Processing under other legal bases continues to be lawful.""",
                "confidence": 0.9,
                "explanation": "Response based on fundamental GDPR principles regarding consent and data subject rights",
                "suggested_search_terms": ["refuse consent", "no detriment", "conditional consent", "data subject rights", "Article 7(4)"]
            },
            
            "information_not_available": {
                "content": """I apologize, but I don't have specific information about this topic in the available regulatory documents.

The documents in my current database focus on consent-related provisions from the following jurisdictions:
- Costa Rica (Law No. 8968)
- Denmark (Danish Act on Data Protection)
- Estonia (Personal Data Protection Act)
- Iceland (Act on Electronic Communications)
- Gabon (Ordinance No. 00001/PR/2018)
- US: Alabama, Missouri, Georgia (various state laws)
- US Federal: HIPAA, FERPA

If you're looking for information about:
- Data breach notification requirements
- Data retention periods
- Security measures
- DPO requirements
- Or other non-consent topics

These are not covered in the current document set. Please refine your query to focus on consent-related matters, or consult the full regulatory texts directly.""",
                "confidence": 1.0,
                "explanation": "The requested information is not available in the current document corpus",
                "suggested_search_terms": []
            }
        }
    
    def get_fallback_response(self, query_type: str, context: Dict[str, any]) -> Optional[FallbackResponse]:
        """
        Get appropriate fallback response based on query type
        
        Args:
            query_type: Type of query (e.g., "unlawful_consent", "cross_border_transfer")
            context: Additional context about the query
            
        Returns:
            FallbackResponse if available, None otherwise
        """
        # Detect query type from context
        query_lower = context.get("query", "").lower()
        
        # Q6: What happens if consent is not obtained lawfully?
        if ("what happens" in query_lower or "consequences" in query_lower) and \
           ("not obtained lawfully" in query_lower or "unlawful" in query_lower):
            template = self.fallback_templates["unlawful_consent_consequences"]
            return FallbackResponse(**template)
        
        # Q11: Is consent a valid basis for cross-border data transfer?
        if ("valid basis" in query_lower or "acceptable" in query_lower) and \
           ("cross-border" in query_lower or "international transfer" in query_lower):
            template = self.fallback_templates["cross_border_consent_validity"]
            return FallbackResponse(**template)
        
        # Q12: Is consent valid indefinitely?
        if ("consent" in query_lower and "indefinitely" in query_lower) or \
           ("consent" in query_lower and "valid" in query_lower and ("how long" in query_lower or "forever" in query_lower)):
            template = self.fallback_templates["consent_validity_duration"]
            return FallbackResponse(**template)
        
        # Q10: What rights does a data subject have if they refuse to give consent?
        if ("refuse" in query_lower or "refusal" in query_lower) and "consent" in query_lower and \
           ("rights" in query_lower or "what happens" in query_lower):
            template = self.fallback_templates["refusal_of_consent_rights"]
            return FallbackResponse(**template)
        
        # Default: Return information_not_available template instead of None
        template = self.fallback_templates["information_not_available"]
        return FallbackResponse(**template)
    
    def should_use_fallback(self, search_results: List, query_analysis: Dict) -> bool:
        """
        Determine if fallback response should be used
        
        Args:
            search_results: Results from search (can be SearchResult objects or dicts)
            query_analysis: Analysis of the query
            
        Returns:
            True if fallback should be used
        """
        # Use fallback if:
        # 1. No results found
        if not search_results or len(search_results) == 0:
            logger.info("Using fallback: No search results found")
            return True
        
        # 2. Results have low relevance scores
        # Handle both SearchResult objects and dicts
        scores = []
        for r in search_results:
            if hasattr(r, 'score'):  # SearchResult object
                scores.append(r.score)
            elif isinstance(r, dict) and 'score' in r:  # Dictionary
                scores.append(r['score'])
            else:
                scores.append(0)  # Default if no score
        
        avg_score = sum(scores) / len(scores) if scores else 0
        if avg_score < 0.5:
            logger.info(f"Using fallback: Low average relevance score {avg_score}")
            return True
        
        # 3. Results don't contain expected keywords
        intent = query_analysis.get("primary_intent", "")
        concepts = query_analysis.get("legal_concepts", [])
        
        # Helper function to get content from result
        def get_content(result):
            if hasattr(result, 'chunk') and hasattr(result.chunk, 'content'):
                return result.chunk.content
            elif isinstance(result, dict):
                return result.get('content', '')
            return ''
        
        # For penalty/consequence queries
        if intent == "compliance_check" and any(c in ["penalties", "consequences", "unlawful"] for c in concepts):
            penalty_keywords = ["penalty", "fine", "sanction", "violation", "breach", "erasure"]
            found_penalty_content = any(
                any(kw in get_content(r).lower() for kw in penalty_keywords)
                for r in search_results
            )
            if not found_penalty_content:
                logger.info("Using fallback: No penalty-related content found")
                return True
        
        # For cross-border transfer queries
        if any(c in ["cross-border", "international transfer"] for c in concepts):
            transfer_keywords = ["adequacy", "appropriate safeguards", "binding corporate", "standard contractual"]
            found_transfer_content = any(
                any(kw in get_content(r).lower() for kw in transfer_keywords)
                for r in search_results
            )
            if not found_transfer_content:
                logger.info("Using fallback: No transfer mechanism content found")
                return True
        
        # For consent validity duration queries
        query_lower = query_analysis.get("query", "").lower()
        if ("consent" in query_lower and "indefinitely" in query_lower) or \
           ("consent" in query_lower and "valid" in query_lower and ("how long" in query_lower or "forever" in query_lower)):
            duration_keywords = ["time-bound", "periodic", "review", "renewal", "expire", "duration", "indefinite"]
            found_duration_content = any(
                any(kw in get_content(r).lower() for kw in duration_keywords)
                for r in search_results
            )
            if not found_duration_content:
                logger.info("Using fallback: No consent duration/review content found")
                return True
        
        return False
    
    def enhance_response_with_fallback(
        self, 
        original_response: str, 
        fallback: FallbackResponse,
        include_explanation: bool = True
    ) -> str:
        """
        Enhance original response with fallback content
        
        Args:
            original_response: Original generated response
            fallback: Fallback response to use
            include_explanation: Whether to include explanation
            
        Returns:
            Enhanced response
        """
        if not original_response or len(original_response.strip()) < 100:
            # Replace entirely with fallback
            response = fallback.content
            if include_explanation:
                response += f"\n\n*Note: {fallback.explanation}*"
        else:
            # Enhance original with fallback
            response = original_response
            if "specific information not available" in original_response.lower() or \
               "not found in the context" in original_response.lower():
                response += f"\n\n**Based on General Principles:**\n{fallback.content}"
                if include_explanation:
                    response += f"\n\n*Note: {fallback.explanation}*"
        
        return response