"""
Integration guide for TermNormalizer with QueryManager and ResponseGenerator

This script shows how to integrate the TermNormalizer service into existing components
to fix the Q2 issue where Costa Rica doesn't appear for "explicit consent" queries.
"""

print("""
INTEGRATION GUIDE: TermNormalizer
================================

To fix the Q2 issue (Costa Rica not appearing for "explicit consent"), integrate 
the TermNormalizer into your services as follows:

1. In QueryManager (src/services/query_manager.py):
   
   # Add import at the top
   from src.services.term_normalizer import get_term_normalizer
   
   # In __init__ method:
   self.term_normalizer = get_term_normalizer()
   
   # In analyze_query method, after extracting legal_concepts:
   # Expand legal concepts to include equivalents
   if legal_concepts:
       expanded_concepts = []
       for concept in legal_concepts:
           equivalents = self.term_normalizer.get_equivalents(concept)
           expanded_concepts.extend(equivalents)
       legal_concepts = list(set(expanded_concepts))  # Remove duplicates

2. In EnhancedRetriever (src/services/enhanced_retriever_service.py):
   
   # Add import at the top
   from src.services.term_normalizer import get_term_normalizer
   
   # In __init__ method:
   self.term_normalizer = get_term_normalizer()
   
   # In _build_search_filter method, when building filter expressions:
   # Expand terms in the search filter
   if search_filter and 'search' in search_filter:
       original_terms = search_filter['search'].split(' OR ')
       expanded_terms = []
       for term in original_terms:
           term_clean = term.strip().strip('"')
           equivalents = self.term_normalizer.get_equivalents(term_clean)
           expanded_terms.extend([f'"{eq}"' for eq in equivalents])
       search_filter['search'] = ' OR '.join(set(expanded_terms))

3. In ResponseGenerator (src/services/response_generator.py):
   
   # Add import at the top
   from src.services.term_normalizer import get_term_normalizer
   
   # In __init__ method:
   self.term_normalizer = get_term_normalizer()
   
   # When processing query terms for matching:
   # Use term equivalency checking
   def _is_term_match(self, doc_term: str, query_term: str) -> bool:
       return self.term_normalizer.is_equivalent(doc_term, query_term)

4. For Simple Search (src/api/endpoints/simple_search.py):
   
   # When building search queries, expand the terms:
   normalizer = get_term_normalizer()
   
   # Original query: "explicit consent"
   # Expanded: "explicit consent" OR "express consent" OR "affirmative consent" ...
   expanded_query = normalizer.expand_query_terms(request.query)

Example Fix for Q2:
==================
Query: "What are the requirements for explicit consent to process sensitive data?"

Before fix:
- Only finds: Andorra, Argentina, Australia, etc.
- Missing: Costa Rica (uses "express consent")

After fix:
- Finds all countries including Costa Rica
- Because "explicit consent" now matches "express consent"

Testing the Fix:
===============
1. Run Q2 query before integration - note missing countries
2. Apply the integration changes above
3. Run Q2 query again - Costa Rica should now appear
4. Verify other equivalencies work (e.g., "sensitive data" = "special categories of data")
""")

# Demonstrate the fix with a simple example
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.term_normalizer import get_term_normalizer

normalizer = get_term_normalizer()

print("\n\nDEMONSTRATION:")
print("=" * 50)

# Show how Costa Rica's "express consent" matches "explicit consent"
query_term = "explicit consent"
costa_rica_term = "express consent"

print(f"\nQuery uses: '{query_term}'")
print(f"Costa Rica uses: '{costa_rica_term}'")
print(f"Are they equivalent? {normalizer.is_equivalent(query_term, costa_rica_term)}")

# Show the expanded search
print(f"\nExpanded search terms for '{query_term}':")
equivalents = normalizer.get_equivalents(query_term)
for eq in equivalents:
    print(f"  - {eq}")

print("\nThis means Costa Rica will now be found in searches for 'explicit consent'!")