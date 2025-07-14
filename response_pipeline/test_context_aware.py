#!/usr/bin/env python3
"""
Test Context-Aware Q&A Functionality
Simple script to test if follow-up detection is working
"""
import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

async def test_context_aware():
    """Test context-aware Q&A with follow-up detection"""
    print("🧪 Testing Context-Aware Q&A")
    print("=" * 40)
    
    try:
        from pipeline.reg_conversational_interface import RegulatoryChatInterface
        
        print("🔧 Initializing chat interface...")
        chat = RegulatoryChatInterface()
        await chat.initialize()
        
        # Check if context manager is available
        if chat.context_manager is None:
            print("⚠️ Context manager not available - context-aware features disabled")
            print("   This is expected if there are import issues")
            print("   The chatbot will still work, just without follow-up detection")
            return False
        else:
            print("✅ Context manager initialized successfully")
        
        # Test scenario: First question + follow-up
        print("\n📝 Test Scenario:")
        print("   Question 1: 'Describe three common errors organizations make about consent'")
        print("   Question 2: 'Give me couple more' (should be detected as follow-up)")
        
        # First query
        print("\n🔄 Processing first query...")
        result1 = await chat.process_query('Describe three common errors organizations make about consent')
        print(f"✅ First query completed in {result1['elapsed_time']:.2f}s")
        
        # Follow-up query
        print("\n🔄 Processing follow-up query...")
        result2 = await chat.process_query('Give me couple more')
        
        # Check results
        context_info = result2.get('context_info', {})
        if context_info.get('is_followup'):
            print(f"✅ SUCCESS: Follow-up detected!")
            print(f"   Confidence: {context_info.get('followup_confidence', 0):.2f}")
            print(f"   Expanded query: {context_info.get('expanded_query', 'N/A')}")
            print(f"   Processing time: {context_info.get('processing_time_ms', 0):.1f}ms")
            return True
        else:
            print(f"❌ FAILED: Follow-up not detected")
            print(f"   Context info: {context_info}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    try:
        success = asyncio.run(test_context_aware())
        
        print("\n" + "=" * 40)
        if success:
            print("🎉 Context-aware Q&A is working correctly!")
            print("   Follow-up detection is functional")
        else:
            print("⚠️ Context-aware features not working")
            print("   Check setup_project.py output for issues")
            
        return success
    except KeyboardInterrupt:
        print("\n⏸️ Test interrupted by user")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1)