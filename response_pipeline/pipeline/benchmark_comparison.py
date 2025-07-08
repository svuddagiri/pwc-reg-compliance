#!/usr/bin/env python3
"""
Benchmark Comparison Tool for Andrea Questions
Compares our system's output vs GPT's benchmark results
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
import sys
import os
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set DEMO_MODE before loading
os.environ['DEMO_MODE'] = 'true'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.response_generator import ResponseGenerator, GenerationRequest

class BenchmarkComparison:
    """Compare our system's output against GPT benchmark"""
    
    def __init__(self):
        self.benchmark_file = Path(__file__).parent.parent / "tests" / "gpt_benchmark_andrea_questions.txt"
        self.gpt_results = self._parse_gpt_benchmark()
        
        # Initialize services
        self.query_manager = QueryManager()
        self.retriever = EnhancedRetrieverService()
        self.response_generator = ResponseGenerator()
        
    def _parse_gpt_benchmark(self) -> Dict[int, Dict]:
        """Parse GPT benchmark file into structured data"""
        
        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by question separators (using the actual separator pattern)
        questions = content.split('------------------------------------------------')
        if len(questions) < 4:
            # Try alternative split pattern
            questions = content.split('--------------------------------------------------------------')
        if len(questions) < 4:
            # Split by question numbers
            parts = re.split(r'\n(?=\d+\.\s)', content)
            questions = parts[1:5] if len(parts) > 4 else parts  # Skip intro, take first 4
        
        results = {}
        
        for i, q_content in enumerate(questions, 1):
            if not q_content.strip():
                continue
                
            lines = q_content.strip().split('\n')
            question_line = lines[0] if lines else ""
            
            # Extract question text
            question_match = re.match(r'^\d+\.\s*(.*)', question_line)
            question = question_match.group(1) if question_match else question_line
            
            # Extract exact quoted text from tables
            exact_quotes = self._extract_exact_quotes(q_content)
            
            # Extract jurisdictions mentioned
            jurisdictions = self._extract_jurisdictions(q_content)
            
            # Extract article/section references
            articles = self._extract_articles(q_content)
            
            # Extract source documents
            sources = self._extract_sources(q_content)
            
            results[i] = {
                'question': question,
                'full_content': q_content,
                'exact_quotes': exact_quotes,
                'jurisdictions': jurisdictions,
                'articles': articles,
                'sources': sources,
                'content_length': len(q_content),
                'has_table_format': self._has_table_format(q_content)
            }
            
        return results
    
    def _extract_exact_quotes(self, content: str) -> List[Dict]:
        """Extract exact quoted regulatory text from GPT response"""
        quotes = []
        
        # Pattern for quoted text in tables
        quote_patterns = [
            r'"([^"]+)"',  # Text in double quotes
            r'Art\.\s*\d+[^"]*"([^"]+)"',  # Article references with quotes
            r'§\s*\d+[^"]*"([^"]+)"',  # Section references with quotes
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if len(match.strip()) > 10:  # Only substantial quotes
                    quotes.append({
                        'text': match.strip(),
                        'length': len(match.strip())
                    })
        
        return quotes
    
    def _extract_jurisdictions(self, content: str) -> List[str]:
        """Extract jurisdictions mentioned in response"""
        jurisdictions = set()
        
        # Core jurisdiction patterns
        patterns = [
            r'\b(Costa Rica|Denmark|Estonia|Georgia|Gabon)\b',
            r'\b(EU|European Union|EEA|GDPR)\b',
            r'\b(United States|US|USA)\b',
            r'\b([A-Z][a-z]+)\s+(?:Act|Law|Regulation|Ordinance)\b',
            r'\b(or states|the act|eea states)\b',  # Handle GPT's specific phrases
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    jurisdictions.update([m for m in match if m.strip()])
                else:
                    jurisdictions.add(match.strip())
        
        # Normalize common variations
        normalized = set()
        for j in jurisdictions:
            j_lower = j.lower()
            if j_lower in ['eu', 'european union', 'gdpr']:
                normalized.add('EU')
            elif j_lower == 'eea states':
                normalized.add('EEA states')
            elif j_lower == 'or states':
                normalized.add('or states')
            elif j_lower == 'the act':
                normalized.add('The Act')
            else:
                normalized.add(j)
        
        return sorted(list(normalized))
    
    def _extract_articles(self, content: str) -> List[str]:
        """Extract article/section references in both short and long format"""
        articles = set()
        
        # GPT's short format patterns (what we're comparing against)
        short_patterns = [
            r'\bArt\.\s*\d+(?:\s*\([^)]+\))?(?:\s*\([^)]+\))?',  # Art. 5, Art. 9(2)
            r'\bArticle\s+\d+(?:\s*\([^)]+\))?',  # Article 9(1)
            r'§\s*\d+(?:\s*\([^)]+\))?',  # § 8(3), § 10(2)
            r'O\.C\.G\.A\.\s*§\s*[\d-]+(?:\s*\([^)]+\))?',  # O.C.G.A. § 20-2-666(a)
        ]
        
        # Our long format patterns - extract and convert to short format
        long_patterns = [
            (r'\[.*?Article\s+(\d+(?:\s*\([^)]+\))?)\]', 'Art. {}'),  # [... Article 5] -> Art. 5
            (r'\[.*?Section\s+(\d+(?:\s*\([^)]+\))?)\]', '§ {}'),     # [... Section 7] -> § 7
            (r'\[.*?Art\.\s*(\d+(?:\s*\([^)]+\))?)\]', 'Art. {}'),   # [... Art. 5] -> Art. 5
            (r'\[.*?§\s*(\d+(?:\s*\([^)]+\))?)\]', '§ {}'),          # [... § 8] -> § 8
        ]
        
        # Extract short format directly
        for pattern in short_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            articles.update(matches)
        
        # Extract from long format and convert to short
        for pattern, format_template in long_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                articles.add(format_template.format(match))
        
        return sorted(list(articles))
    
    def _extract_sources(self, content: str) -> List[str]:
        """Extract source document references from GPT response"""
        sources = set()
        
        patterns = [
            r'Regulation\s+No\.\s*[\d-]+',
            r'Act\s+No\.\s*[\d/]+',
            r'Ordinance\s+No\.\s*[\d/]+',
            r'Executive\s+Decree\s+No\.\s*[\d-]+',
            r'Personal\s+Data\s+Protection\s+Act',
            r'Student\s+Data\s+Privacy\s+Act'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            sources.update(matches)
            
        return sorted(list(sources))
    
    def _has_table_format(self, content: str) -> bool:
        """Check if content has structured table format like GPT"""
        table_indicators = [
            'Jurisdiction\t',
            '|\t',
            'Document name\t',
            'Exact text',
            'Article / section\t'
        ]
        
        return any(indicator in content for indicator in table_indicators)
    
    async def test_our_system(self, question: str) -> Dict:
        """Test our system with a question and return structured results"""
        start_time = time.time()
        
        try:
            # Step 1: Analyze query
            query_analysis = await self.query_manager.analyze_query(question)
            
            # Step 2: Retrieve chunks
            search_results = await self.retriever.retrieve(query_analysis=query_analysis)
            
            # Step 3: Generate response
            generation_request = GenerationRequest(
                user_id=13,
                session_id="benchmark_test",
                conversation_id=1,
                message_id=1,
                query=question,
                query_analysis=query_analysis,
                search_results=search_results.results,
                conversation_history=[],
                stream=False,
                model="gpt-4",
                temperature=0.0,
                max_tokens=2000
            )
            
            response = await self.response_generator.generate(generation_request)
            end_time = time.time()
            
            # Extract analysis from our response
            content = response.content
            
            return {
                'success': True,
                'content': content,
                'content_length': len(content),
                'response_time': end_time - start_time,
                'citations': getattr(response, 'citations', []),
                'citation_count': len(getattr(response, 'citations', [])),
                'exact_quotes': self._extract_exact_quotes(content),
                'jurisdictions': self._extract_jurisdictions(content),
                'articles': self._extract_articles(content),
                'sources': self._extract_sources(content),
                'has_table_format': self._has_table_format(content),
                'chunks_retrieved': len(search_results.results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def compare_responses(self, our_result: Dict, gpt_result: Dict) -> Dict:
        """Compare our response vs GPT benchmark"""
        if not our_result.get('success', False):
            return {
                'overall_score': 0,
                'error': our_result.get('error', 'Unknown error'),
                'comparison': {}
            }
        
        comparison = {}
        
        # 1. Content similarity (semantic)
        our_content = our_result['content']
        gpt_content = gpt_result['full_content']
        content_similarity = SequenceMatcher(None, our_content.lower(), gpt_content.lower()).ratio()
        
        # 2. Exact quotes comparison
        our_quotes = our_result['exact_quotes']
        gpt_quotes = gpt_result['exact_quotes']
        
        quote_matches = 0
        if gpt_quotes:
            for gpt_quote in gpt_quotes:
                for our_quote in our_quotes:
                    similarity = SequenceMatcher(None, 
                                               our_quote['text'].lower(), 
                                               gpt_quote['text'].lower()).ratio()
                    if similarity > 0.8:  # 80% similarity threshold
                        quote_matches += 1
                        break
            quote_score = quote_matches / len(gpt_quotes) if gpt_quotes else 0
        else:
            quote_score = 1.0  # If GPT has no quotes, we don't penalize
        
        # 3. Jurisdiction coverage
        our_jurisdictions = set(j.lower() for j in our_result['jurisdictions'])
        gpt_jurisdictions = set(j.lower() for j in gpt_result['jurisdictions'])
        
        if gpt_jurisdictions:
            jurisdiction_overlap = len(our_jurisdictions & gpt_jurisdictions)
            jurisdiction_score = jurisdiction_overlap / len(gpt_jurisdictions)
        else:
            jurisdiction_score = 1.0
        
        # 4. Article reference accuracy
        our_articles = set(our_result['articles'])
        gpt_articles = set(gpt_result['articles'])
        
        if gpt_articles:
            article_overlap = len(our_articles & gpt_articles)
            article_score = article_overlap / len(gpt_articles)
        else:
            article_score = 1.0
        
        # 5. Structure/format comparison
        our_has_table = our_result['has_table_format']
        gpt_has_table = gpt_result['has_table_format']
        structure_score = 1.0 if (our_has_table == gpt_has_table) else 0.5
        
        # 6. Content completeness (length comparison)
        length_ratio = min(our_result['content_length'] / gpt_result['content_length'], 1.0) if gpt_result['content_length'] > 0 else 0
        
        # Calculate overall score (weighted)
        weights = {
            'content_similarity': 0.25,
            'quote_accuracy': 0.30,  # High weight for exact quotes
            'jurisdiction_coverage': 0.20,
            'article_accuracy': 0.15,
            'structure': 0.05,
            'completeness': 0.05
        }
        
        scores = {
            'content_similarity': content_similarity,
            'quote_accuracy': quote_score,
            'jurisdiction_coverage': jurisdiction_score,
            'article_accuracy': article_score,
            'structure': structure_score,
            'completeness': length_ratio
        }
        
        overall_score = sum(scores[k] * weights[k] for k in weights.keys())
        
        comparison = {
            'overall_score': overall_score,
            'detailed_scores': scores,
            'metrics': {
                'our_quote_count': len(our_quotes),
                'gpt_quote_count': len(gpt_quotes),
                'quote_matches': quote_matches,
                'our_jurisdiction_count': len(our_result['jurisdictions']),
                'gpt_jurisdiction_count': len(gpt_result['jurisdictions']),
                'jurisdiction_overlap': len(our_jurisdictions & gpt_jurisdictions),
                'our_article_count': len(our_result['articles']),
                'gpt_article_count': len(gpt_result['articles']),
                'article_overlap': len(our_articles & gpt_articles),
                'content_length_ratio': length_ratio
            },
            'missing_jurisdictions': list(gpt_jurisdictions - our_jurisdictions),
            'missing_articles': list(gpt_articles - our_articles),
            'performance': {
                'response_time': our_result['response_time'],
                'chunks_retrieved': our_result.get('chunks_retrieved', 0)
            }
        }
        
        return comparison
    
    async def run_full_benchmark(self) -> Dict:
        """Run benchmark comparison for all 4 questions"""
        print("🚀 Starting Benchmark Comparison vs GPT")
        print("=" * 60)
        
        results = {}
        total_start = time.time()
        
        for q_num in range(1, 5):
            if q_num not in self.gpt_results:
                print(f"❌ Q{q_num}: GPT benchmark data not found")
                continue
                
            gpt_data = self.gpt_results[q_num]
            question = gpt_data['question']
            
            print(f"\n🔍 Testing Q{q_num}: {question[:60]}...")
            
            # Test our system
            our_result = await self.test_our_system(question)
            
            if our_result['success']:
                print(f"   ✅ Our system: {our_result['response_time']:.1f}s, {our_result['content_length']} chars")
                print(f"   📚 Citations: {our_result['citation_count']}, Chunks: {our_result.get('chunks_retrieved', 0)}")
                print(f"   🌍 Jurisdictions found: {len(our_result['jurisdictions'])}")
            else:
                print(f"   ❌ Our system failed: {our_result.get('error', 'Unknown error')}")
            
            # Compare vs GPT
            comparison = self.compare_responses(our_result, gpt_data)
            
            if 'overall_score' in comparison:
                score = comparison['overall_score']
                print(f"   📊 Benchmark Score: {score:.1%}")
                
                # Show key gaps
                if comparison.get('missing_jurisdictions'):
                    print(f"   ⚠️  Missing jurisdictions: {', '.join(comparison['missing_jurisdictions'])}")
                
                if comparison.get('missing_articles'):
                    print(f"   ⚠️  Missing articles: {', '.join(comparison['missing_articles'])}")
                
                metrics = comparison.get('metrics', {})
                if metrics.get('quote_matches', 0) < metrics.get('gpt_quote_count', 0):
                    print(f"   ⚠️  Quote accuracy: {metrics.get('quote_matches', 0)}/{metrics.get('gpt_quote_count', 0)} exact quotes matched")
            
            results[q_num] = {
                'question': question,
                'our_result': our_result,
                'gpt_result': gpt_data,
                'comparison': comparison
            }
            
            # Brief pause between questions
            await asyncio.sleep(1)
        
        total_time = time.time() - total_start
        
        # Generate summary
        summary = self._generate_summary(results, total_time)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_comparison_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_time': total_time,
                'summary': summary,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        return results, summary
    
    def _generate_summary(self, results: Dict, total_time: float) -> Dict:
        """Generate summary of benchmark comparison"""
        successful_tests = [r for r in results.values() if r['our_result'].get('success', False)]
        
        if not successful_tests:
            return {
                'overall_performance': 'FAILED',
                'success_rate': '0%',
                'total_time': total_time,
                'issues': ['All questions failed']
            }
        
        # Calculate averages
        avg_score = sum(r['comparison'].get('overall_score', 0) for r in successful_tests) / len(successful_tests)
        avg_time = sum(r['our_result']['response_time'] for r in successful_tests) / len(successful_tests)
        
        # Identify key issues
        issues = []
        recommendations = []
        
        for q_num, result in results.items():
            if not result['our_result'].get('success', False):
                issues.append(f"Q{q_num} failed to complete")
                continue
                
            comparison = result['comparison']
            scores = comparison.get('detailed_scores', {})
            
            # Check specific issue patterns
            if scores.get('quote_accuracy', 0) < 0.5:
                issues.append(f"Q{q_num}: Poor exact quote accuracy ({scores.get('quote_accuracy', 0):.1%})")
                recommendations.append("Improve exact text extraction - stop paraphrasing regulatory text")
            
            if scores.get('jurisdiction_coverage', 0) < 0.7:
                missing = comparison.get('missing_jurisdictions', [])
                issues.append(f"Q{q_num}: Missing jurisdictions ({', '.join(missing[:3])})")
                recommendations.append("Fix jurisdiction detection and inclusion logic")
            
            if scores.get('article_accuracy', 0) < 0.5:
                issues.append(f"Q{q_num}: Missing article references")
                recommendations.append("Improve citation extraction with exact article numbers")
            
            if result['our_result']['response_time'] > 10:
                issues.append(f"Q{q_num}: Slow response ({result['our_result']['response_time']:.1f}s)")
                recommendations.append("Optimize performance - target <5s response time")
        
        # Overall assessment
        if avg_score >= 0.8:
            performance = "EXCELLENT"
        elif avg_score >= 0.6:
            performance = "GOOD"
        elif avg_score >= 0.4:
            performance = "FAIR"
        else:
            performance = "POOR"
        
        return {
            'overall_performance': performance,
            'average_benchmark_score': f"{avg_score:.1%}",
            'success_rate': f"{len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.0f}%)",
            'average_response_time': f"{avg_time:.1f}s",
            'total_time': f"{total_time:.1f}s",
            'key_issues': issues[:5],  # Top 5 issues
            'recommendations': list(set(recommendations))[:5],  # Top 5 unique recommendations
            'detailed_scores': {
                'quote_accuracy': f"{sum(r['comparison'].get('detailed_scores', {}).get('quote_accuracy', 0) for r in successful_tests) / len(successful_tests):.1%}",
                'jurisdiction_coverage': f"{sum(r['comparison'].get('detailed_scores', {}).get('jurisdiction_coverage', 0) for r in successful_tests) / len(successful_tests):.1%}",
                'article_accuracy': f"{sum(r['comparison'].get('detailed_scores', {}).get('article_accuracy', 0) for r in successful_tests) / len(successful_tests):.1%}",
                'content_similarity': f"{sum(r['comparison'].get('detailed_scores', {}).get('content_similarity', 0) for r in successful_tests) / len(successful_tests):.1%}"
            }
        }

async def main():
    """Run the benchmark comparison"""
    comparator = BenchmarkComparison()
    
    print("📋 GPT Benchmark Analysis:")
    print(f"   Questions loaded: {len(comparator.gpt_results)}")
    for q_num, data in comparator.gpt_results.items():
        print(f"   Q{q_num}: {len(data['exact_quotes'])} exact quotes, {len(data['jurisdictions'])} jurisdictions")
    
    try:
        results, summary = await comparator.run_full_benchmark()
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"🎯 Overall Performance: {summary['overall_performance']}")
        print(f"📈 Average Benchmark Score: {summary['average_benchmark_score']}")
        print(f"✅ Success Rate: {summary['success_rate']}")
        print(f"⏱️  Average Response Time: {summary['average_response_time']}")
        
        print(f"\n📊 Detailed Scores:")
        for metric, score in summary['detailed_scores'].items():
            print(f"   {metric.replace('_', ' ').title()}: {score}")
        
        if summary['key_issues']:
            print(f"\n⚠️  Key Issues:")
            for issue in summary['key_issues']:
                print(f"   - {issue}")
        
        if summary['recommendations']:
            print(f"\n🎯 Priority Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())