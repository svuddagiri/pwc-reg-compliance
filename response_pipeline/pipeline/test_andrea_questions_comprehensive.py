#!/usr/bin/env python3
"""
Comprehensive test script for Andrea's 4 questions
Runs all questions, captures detailed results, and provides analysis
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import os
from typing import Dict, List, Any
import re

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.conversation_manager import ConversationManager
from src.models.chat import (
    ChatRequest, 
    GenerationRequest, 
    QueryAnalysis, 
    UserInfo
)

class AndreaQuestionsAnalyzer:
    """Comprehensive analyzer for Andrea's questions"""
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.questions = [
            "Which countries or states require consent opt-in consent for processing of sensitive personal information?",
            "Which countries or states have requirements around obtaining consent from parents or guardians for processing data of minors (under 18 years of age)?",
            "When listing the requirements for valid consent for data processing, what requirements are most common across the different regulations?",
            "Retrieve and summarize all the definitions/considerations for affirmative consent for all the sources within the knowledge base."
        ]
        self.results = []
        
    async def run_question(self, question: str, question_num: int) -> Dict[str, Any]:
        """Run a single question and capture detailed results"""
        print(f"\n🔍 Testing Q{question_num}: {question}")
        
        start_time = time.time()
        
        try:
            # Create request
            user_info = UserInfo(
                user_id="test_user_andrea",
                session_id=f"andrea_test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name="Andrea Test User"
            )
            
            chat_request = ChatRequest(
                message=question,
                conversation_id=f"andrea_conv_{question_num}",
                message_id=f"msg_{question_num}",
                user_info=user_info,
                model="gpt-4",
                temperature=0.0,
                max_tokens=2000
            )
            
            # Process the question
            response = await self.conversation_manager.process_message(chat_request)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract key information
            result = {
                "question_number": question_num,
                "question": question,
                "response_time_seconds": round(response_time, 2),
                "success": True,
                "response": {
                    "content": response.content if hasattr(response, 'content') else str(response),
                    "citations": getattr(response, 'citations', []),
                    "confidence_score": getattr(response, 'confidence_score', None),
                    "tokens_used": getattr(response, 'tokens_used', None),
                    "model_used": getattr(response, 'model_used', None)
                },
                "analysis": {
                    "content_length": len(response.content if hasattr(response, 'content') else str(response)),
                    "citation_count": len(getattr(response, 'citations', [])),
                    "jurisdictions_mentioned": self._extract_jurisdictions(response.content if hasattr(response, 'content') else str(response)),
                    "key_concepts_found": self._extract_concepts(response.content if hasattr(response, 'content') else str(response)),
                    "response_quality": self._assess_quality(question, response.content if hasattr(response, 'content') else str(response))
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Q{question_num} completed in {response_time:.1f}s")
            print(f"   📄 Content: {result['analysis']['content_length']} chars")
            print(f"   📚 Citations: {result['analysis']['citation_count']}")
            print(f"   🌍 Jurisdictions: {len(result['analysis']['jurisdictions_mentioned'])}")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"❌ Q{question_num} failed after {response_time:.1f}s: {str(e)}")
            
            return {
                "question_number": question_num,
                "question": question,
                "response_time_seconds": round(response_time, 2),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_jurisdictions(self, content: str) -> List[str]:
        """Extract jurisdiction names from response content"""
        jurisdiction_patterns = [
            r'\b(Denmark|Estonia|Costa Rica|Iceland|Georgia|Gabon|Missouri|Alabama|US|United States|California|GDPR|European Union|EU)\b',
            r'\b([A-Z][a-z]+ (?:Act|Law|Regulation|Ordinance))\b',
            r'\[(.*?)\s+(?:Article|Section|§)\s+\d+\]'
        ]
        
        jurisdictions = set()
        for pattern in jurisdiction_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    jurisdictions.update([m for m in match if m.strip()])
                else:
                    jurisdictions.add(match)
        
        return sorted(list(jurisdictions))
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key consent-related concepts"""
        concepts = []
        concept_patterns = {
            "explicit_consent": r'\b(?:explicit|express|clear|unambiguous)\s+consent\b',
            "informed_consent": r'\binformed\s+consent\b',
            "specific_consent": r'\bspecific\s+consent\b',
            "freely_given": r'\bfreely\s+given\b',
            "withdrawal": r'\b(?:withdraw|revoke|revocation)\b',
            "minors": r'\b(?:minors|children|under\s+18|parental)\b',
            "sensitive_data": r'\b(?:sensitive|special|biometric|health)\s+(?:data|information)\b',
            "affirmative_consent": r'\b(?:affirmative|opt-in|positive)\s+consent\b'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                concepts.append(concept)
        
        return concepts
    
    def _assess_quality(self, question: str, content: str) -> Dict[str, Any]:
        """Assess response quality based on question type"""
        quality = {
            "addresses_question": False,
            "provides_specifics": False,
            "includes_citations": False,
            "comprehensive": False,
            "issues": []
        }
        
        # Check if it addresses the question
        if "jurisdiction" in question.lower() or "countries" in question.lower():
            if any(j in content for j in ["Denmark", "Estonia", "Costa Rica", "GDPR", "California"]):
                quality["addresses_question"] = True
            else:
                quality["issues"].append("Missing expected jurisdictions")
        
        # Check for specific requirements/definitions
        if "requirements" in question.lower() or "definitions" in question.lower():
            requirement_indicators = ["must", "shall", "required", "necessary", "Article", "Section"]
            if any(req in content for req in requirement_indicators):
                quality["provides_specifics"] = True
            else:
                quality["issues"].append("Lacks specific requirements or definitions")
        
        # Check for citations
        citation_patterns = [r'\[.*?\]', r'Article\s+\d+', r'Section\s+\d+', r'§\s*\d+']
        if any(re.search(pattern, content) for pattern in citation_patterns):
            quality["includes_citations"] = True
        else:
            quality["issues"].append("Missing proper citations")
        
        # Check comprehensiveness (length and detail)
        if len(content) > 500 and len(content.split()) > 100:
            quality["comprehensive"] = True
        else:
            quality["issues"].append("Response too brief")
        
        # Check for error indicators
        error_indicators = ["no information", "not available", "cannot provide", "error", "failed"]
        if any(error in content.lower() for error in error_indicators):
            quality["issues"].append("Contains error or unavailability statements")
        
        return quality
    
    async def run_all_questions(self):
        """Run all questions and capture results"""
        print("🚀 Starting Andrea Questions Comprehensive Test")
        print("=" * 60)
        
        overall_start = time.time()
        
        for i, question in enumerate(self.questions, 1):
            result = await self.run_question(question, i)
            self.results.append(result)
            
            # Brief pause between questions
            await asyncio.sleep(1)
        
        overall_time = time.time() - overall_start
        
        # Generate summary
        summary = self._generate_summary(overall_time)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"andrea_questions_results_{timestamp}.json"
        analysis_file = f"andrea_questions_analysis_{timestamp}.txt"
        
        # Save JSON results
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_time_seconds": round(overall_time, 2),
                    "questions_tested": len(self.questions)
                },
                "results": self.results,
                "summary": summary
            }, f, indent=2, ensure_ascii=False)
        
        # Save analysis text
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_analysis_report(summary))
        
        print(f"\n📊 Results saved to: {results_file}")
        print(f"📋 Analysis saved to: {analysis_file}")
        
        return self.results, summary
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive summary of results"""
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]
        
        if successful:
            avg_time = sum(r['response_time_seconds'] for r in successful) / len(successful)
            avg_content_length = sum(r['analysis']['content_length'] for r in successful) / len(successful)
            total_citations = sum(r['analysis']['citation_count'] for r in successful)
            all_jurisdictions = set()
            all_concepts = set()
            
            for r in successful:
                all_jurisdictions.update(r['analysis']['jurisdictions_mentioned'])
                all_concepts.update(r['analysis']['key_concepts_found'])
        else:
            avg_time = 0
            avg_content_length = 0
            total_citations = 0
            all_jurisdictions = set()
            all_concepts = set()
        
        summary = {
            "overall_performance": {
                "total_questions": len(self.questions),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{(len(successful) / len(self.questions)) * 100:.1f}%",
                "total_time_seconds": round(total_time, 2),
                "average_response_time": round(avg_time, 2) if successful else None
            },
            "content_analysis": {
                "average_content_length": round(avg_content_length) if successful else 0,
                "total_citations": total_citations,
                "unique_jurisdictions_found": len(all_jurisdictions),
                "jurisdictions": sorted(list(all_jurisdictions)),
                "key_concepts_coverage": len(all_concepts),
                "concepts_found": sorted(list(all_concepts))
            },
            "quality_assessment": self._assess_overall_quality(),
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _assess_overall_quality(self) -> Dict[str, Any]:
        """Assess overall quality across all responses"""
        successful = [r for r in self.results if r.get('success', False)]
        
        if not successful:
            return {"overall_score": "FAILED", "issues": ["All questions failed"]}
        
        quality_metrics = {
            "addresses_questions": 0,
            "provides_specifics": 0,
            "includes_citations": 0,
            "comprehensive": 0
        }
        
        all_issues = []
        
        for result in successful:
            quality = result['analysis']['response_quality']
            for metric in quality_metrics:
                if quality.get(metric, False):
                    quality_metrics[metric] += 1
            all_issues.extend(quality.get('issues', []))
        
        # Calculate scores
        total_successful = len(successful)
        scores = {k: (v / total_successful) * 100 for k, v in quality_metrics.items()}
        
        # Overall score
        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= 75:
            overall = "GOOD"
        elif avg_score >= 50:
            overall = "FAIR"
        else:
            overall = "POOR"
        
        return {
            "overall_score": overall,
            "average_score": round(avg_score, 1),
            "metric_scores": {k: f"{v:.1f}%" for k, v in scores.items()},
            "common_issues": list(set(all_issues))
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        successful = [r for r in self.results if r.get('success', False)]
        
        if len(successful) < len(self.questions):
            recommendations.append("🚨 CRITICAL: Fix system errors preventing some questions from completing")
        
        if successful:
            avg_time = sum(r['response_time_seconds'] for r in successful) / len(successful)
            if avg_time > 10:
                recommendations.append(f"⚡ PERFORMANCE: Average response time ({avg_time:.1f}s) exceeds target (<5s)")
            
            # Check jurisdiction coverage
            q1_result = next((r for r in successful if r['question_number'] == 1), None)
            if q1_result and len(q1_result['analysis']['jurisdictions_mentioned']) < 3:
                recommendations.append("🌍 COVERAGE: Q1 should identify more jurisdictions (expect Costa Rica, Denmark, Estonia, GDPR)")
            
            # Check citation quality
            avg_citations = sum(r['analysis']['citation_count'] for r in successful) / len(successful)
            if avg_citations < 3:
                recommendations.append("📚 CITATIONS: Responses need more detailed legal citations")
            
            # Check for specific concepts
            all_concepts = set()
            for r in successful:
                all_concepts.update(r['analysis']['key_concepts_found'])
            
            expected_concepts = ["explicit_consent", "informed_consent", "freely_given", "withdrawal"]
            missing_concepts = [c for c in expected_concepts if c not in all_concepts]
            if missing_concepts:
                recommendations.append(f"🔍 CONCEPTS: Missing key concepts: {', '.join(missing_concepts)}")
        
        if not recommendations:
            recommendations.append("✅ EXCELLENT: System performing well across all metrics")
        
        return recommendations
    
    def _generate_analysis_report(self, summary: Dict[str, Any]) -> str:
        """Generate detailed analysis report"""
        report = []
        report.append("=" * 80)
        report.append("ANDREA QUESTIONS COMPREHENSIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall Performance
        perf = summary['overall_performance']
        report.append("📊 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Questions Tested: {perf['total_questions']}")
        report.append(f"Success Rate: {perf['success_rate']}")
        report.append(f"Total Time: {perf['total_time_seconds']}s")
        report.append(f"Average Response Time: {perf['average_response_time']}s" if perf['average_response_time'] else "N/A")
        report.append("")
        
        # Content Analysis
        content = summary['content_analysis']
        report.append("📄 CONTENT ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average Content Length: {content['average_content_length']} characters")
        report.append(f"Total Citations: {content['total_citations']}")
        report.append(f"Unique Jurisdictions: {content['unique_jurisdictions_found']}")
        report.append(f"Jurisdictions Found: {', '.join(content['jurisdictions'])}")
        report.append(f"Key Concepts Coverage: {content['key_concepts_coverage']}")
        report.append(f"Concepts Found: {', '.join(content['concepts_found'])}")
        report.append("")
        
        # Quality Assessment
        quality = summary['quality_assessment']
        report.append("⭐ QUALITY ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Overall Score: {quality['overall_score']}")
        report.append(f"Average Score: {quality['average_score']}%")
        report.append("Metric Breakdown:")
        for metric, score in quality['metric_scores'].items():
            report.append(f"  - {metric.replace('_', ' ').title()}: {score}")
        
        if quality['common_issues']:
            report.append("Common Issues:")
            for issue in quality['common_issues']:
                report.append(f"  ⚠️  {issue}")
        report.append("")
        
        # Recommendations
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(summary['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Individual Question Results
        report.append("🔍 INDIVIDUAL QUESTION ANALYSIS")
        report.append("-" * 40)
        for result in self.results:
            if result.get('success', False):
                report.append(f"Q{result['question_number']}: ✅ SUCCESS ({result['response_time_seconds']}s)")
                analysis = result['analysis']
                report.append(f"   Content: {analysis['content_length']} chars")
                report.append(f"   Citations: {analysis['citation_count']}")
                report.append(f"   Jurisdictions: {len(analysis['jurisdictions_mentioned'])}")
                if analysis['response_quality']['issues']:
                    report.append(f"   Issues: {', '.join(analysis['response_quality']['issues'])}")
            else:
                report.append(f"Q{result['question_number']}: ❌ FAILED ({result['response_time_seconds']}s)")
                report.append(f"   Error: {result.get('error', 'Unknown error')}")
            report.append("")
        
        return "\n".join(report)

async def main():
    """Main function to run the comprehensive test"""
    analyzer = AndreaQuestionsAnalyzer()
    
    try:
        results, summary = await analyzer.run_all_questions()
        
        print("\n" + "=" * 60)
        print("📋 QUICK SUMMARY")
        print("=" * 60)
        
        perf = summary['overall_performance']
        print(f"✅ Success Rate: {perf['success_rate']}")
        print(f"⏱️  Average Time: {perf['average_response_time']}s" if perf['average_response_time'] else "N/A")
        print(f"📚 Total Citations: {summary['content_analysis']['total_citations']}")
        print(f"🌍 Jurisdictions: {summary['content_analysis']['unique_jurisdictions_found']}")
        print(f"⭐ Quality Score: {summary['quality_assessment']['overall_score']}")
        
        print(f"\n🎯 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(summary['recommendations'][:3], 1):
            print(f"{i}. {rec}")
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())