"""
Test script for REST API endpoints
"""
import asyncio
import httpx
import json
from typing import Dict, Any
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Base URL for API
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Test credentials
TEST_USER = {
    "email": "test@example.com",
    "username": "test_user",
    "password": "TestPass123!",
    "full_name": "Test User"
}

ADMIN_USER = {
    "username": "admin@datafactz.com",
    "password": "password123"
}


class APITester:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.access_token = None
        self.admin_token = None
        
    async def close(self):
        await self.client.aclose()
        
    async def test_auth_endpoints(self):
        """Test authentication endpoints"""
        logger.info("Testing authentication endpoints...")
        
        # Test registration
        try:
            response = await self.client.post(
                f"{BASE_URL}/auth/register",
                json=TEST_USER
            )
            if response.status_code == 200:
                logger.info("✅ Registration successful")
            elif response.status_code == 400:
                logger.info("⚠️  User already exists")
            else:
                logger.error(f"❌ Registration failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
        
        # Test login
        try:
            response = await self.client.post(
                f"{BASE_URL}/auth/login",
                data={
                    "username": TEST_USER["username"],
                    "password": TEST_USER["password"]
                }
            )
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                logger.info("✅ Login successful")
                logger.info(f"   Token: {self.access_token[:20]}...")
            else:
                logger.error(f"❌ Login failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
        
        # Test admin login
        try:
            response = await self.client.post(
                f"{BASE_URL}/auth/login",
                data={
                    "username": ADMIN_USER["username"],
                    "password": ADMIN_USER["password"]
                }
            )
            if response.status_code == 200:
                data = response.json()
                self.admin_token = data["access_token"]
                logger.info("✅ Admin login successful")
            else:
                logger.error(f"❌ Admin login failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Admin login error: {e}")
        
        # Test get current user
        if self.access_token:
            try:
                response = await self.client.get(
                    f"{BASE_URL}/auth/me",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                if response.status_code == 200:
                    user_data = response.json()
                    logger.info("✅ Get current user successful")
                    logger.info(f"   User: {user_data['username']} ({user_data['email']})")
                else:
                    logger.error(f"❌ Get current user failed: {response.text}")
            except Exception as e:
                logger.error(f"❌ Get current user error: {e}")
    
    async def test_chat_endpoints(self):
        """Test chat endpoints"""
        if not self.access_token:
            logger.warning("⚠️  No access token, skipping chat tests")
            return
            
        logger.info("\nTesting chat endpoints...")
        
        # Test sending a message
        try:
            response = await self.client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "message": "What is GDPR and when did it come into effect?",
                    "metadata": {"source": "api_test"}
                },
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Send message successful")
                logger.info(f"   Response: {data['content'][:100]}...")
                logger.info(f"   Citations: {len(data['citations'])} found")
                logger.info(f"   Intent: {data['intent']}")
                logger.info(f"   Confidence: {data['confidence_score']}")
                
                # Save conversation ID for further tests
                self.conversation_id = data["conversation_id"]
            else:
                logger.error(f"❌ Send message failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Send message error: {e}")
        
        # Test getting conversations
        try:
            response = await self.client.get(
                f"{BASE_URL}/chat/conversations",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            if response.status_code == 200:
                conversations = response.json()
                logger.info(f"✅ Get conversations successful: {len(conversations)} found")
                for conv in conversations[:3]:  # Show first 3
                    logger.info(f"   - {conv['title'][:50]}... (ID: {conv['conversation_id']})")
            else:
                logger.error(f"❌ Get conversations failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Get conversations error: {e}")
        
        # Test streaming (just initiate, don't wait for full response)
        try:
            logger.info("Testing streaming endpoint...")
            async with self.client.stream(
                "POST",
                f"{BASE_URL}/chat/message/stream",
                json={
                    "message": "Explain CCPA in simple terms",
                    "metadata": {"source": "api_test_stream"}
                },
                headers={"Authorization": f"Bearer {self.access_token}"}
            ) as response:
                if response.status_code == 200:
                    logger.info("✅ Streaming initiated successfully")
                    # Read first few chunks
                    chunk_count = 0
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            chunk_count += 1
                            if chunk_count >= 5:  # Just show we can receive chunks
                                logger.info(f"   Received {chunk_count} chunks...")
                                break
                else:
                    logger.error(f"❌ Streaming failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
    
    async def test_admin_endpoints(self):
        """Test admin endpoints"""
        if not self.admin_token:
            logger.warning("⚠️  No admin token, skipping admin tests")
            return
            
        logger.info("\nTesting admin endpoints...")
        
        # Test usage stats
        try:
            response = await self.client.get(
                f"{BASE_URL}/admin/usage/stats",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if response.status_code == 200:
                stats = response.json()
                logger.info("✅ Get usage stats successful")
                logger.info(f"   Total requests: {stats['total_requests']}")
                logger.info(f"   Total users: {stats['total_users']}")
                logger.info(f"   Total cost: ${stats['total_cost_usd']:.2f}")
            else:
                logger.error(f"❌ Get usage stats failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Get usage stats error: {e}")
        
        # Test model usage
        try:
            response = await self.client.get(
                f"{BASE_URL}/admin/models/usage",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ Get model usage successful: {len(models)} models")
                for model in models:
                    logger.info(f"   - {model['model']}: {model['request_count']} requests")
            else:
                logger.error(f"❌ Get model usage failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Get model usage error: {e}")
        
        # Test system health
        try:
            response = await self.client.get(
                f"{BASE_URL}/admin/system/health",
                headers={"Authorization": f"Bearer {self.admin_token}"}
            )
            if response.status_code == 200:
                health = response.json()
                logger.info("✅ Get system health successful")
                logger.info(f"   Status: {health['status']}")
                logger.info(f"   Database: {health['database_status']}")
                logger.info(f"   Error rate: {health['error_rate_percent']}%")
                logger.info(f"   Active sessions: {health['active_sessions']}")
            else:
                logger.error(f"❌ Get system health failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Get system health error: {e}")
    
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        logger.info("\nTesting health endpoint...")
        
        try:
            response = await self.client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                health = response.json()
                logger.info("✅ Health check successful")
                logger.info(f"   Status: {health['status']}")
                logger.info(f"   Version: {health['version']}")
            else:
                logger.error(f"❌ Health check failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")


async def main():
    """Run all API tests"""
    logger.info("Starting API endpoint tests...")
    logger.info(f"Base URL: {BASE_URL}")
    
    tester = APITester()
    
    try:
        # Run tests in sequence
        await tester.test_health_endpoint()
        await tester.test_auth_endpoints()
        await tester.test_chat_endpoints()
        await tester.test_admin_endpoints()
        
        logger.info("\n✅ API tests completed!")
        
    except Exception as e:
        logger.error(f"Test suite error: {e}")
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())