"""
Test script for Analytics Service
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.analytics_service import AnalyticsService
from src.clients.azure_sql import AzureSQLClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_analytics_service():
    """Test all analytics service functions"""
    logger.info("Testing Analytics Service...")
    
    async with AzureSQLClient() as sql_client:
        async with sql_client.get_session() as session:
            analytics = AnalyticsService(session)
            
            # Test 1: Daily aggregation
            logger.info("\n1. Testing daily aggregation...")
            try:
                yesterday = (datetime.utcnow() - timedelta(days=1)).date()
                stats = await analytics.aggregate_daily_usage(yesterday)
                logger.info(f"✅ Daily aggregation successful for {yesterday}")
                logger.info(f"   Total requests: {stats['total_requests']}")
                logger.info(f"   Unique users: {stats['unique_users']}")
                logger.info(f"   Total cost: ${stats['total_cost_usd']:.2f}")
            except Exception as e:
                logger.error(f"❌ Daily aggregation failed: {e}")
            
            # Test 2: Usage trends
            logger.info("\n2. Testing usage trends...")
            try:
                trends = await analytics.get_usage_trends(days=7)
                logger.info("✅ Usage trends retrieved successfully")
                logger.info(f"   Period: {trends['period']['days']} days")
                logger.info(f"   Total requests: {trends['summary']['total_requests']}")
                logger.info(f"   Request trend: {trends['trends']['requests']:.1f}%")
                logger.info(f"   Cost trend: {trends['trends']['cost']:.1f}%")
            except Exception as e:
                logger.error(f"❌ Usage trends failed: {e}")
            
            # Test 3: Cost breakdown
            logger.info("\n3. Testing cost breakdown...")
            try:
                cost = await analytics.get_cost_breakdown()
                logger.info("✅ Cost breakdown retrieved successfully")
                logger.info(f"   Total cost: ${cost['total_cost']:.2f}")
                logger.info(f"   Daily projection: ${cost['cost_projection']['daily']:.2f}")
                logger.info(f"   Monthly projection: ${cost['cost_projection']['monthly']:.2f}")
                logger.info("   Top models by cost:")
                for model in cost['by_model'][:3]:
                    logger.info(f"     - {model['model']}: ${model['total_cost']:.2f}")
            except Exception as e:
                logger.error(f"❌ Cost breakdown failed: {e}")
            
            # Test 4: Anomaly detection
            logger.info("\n4. Testing anomaly detection...")
            try:
                anomalies = await analytics.detect_anomalies()
                logger.info(f"✅ Anomaly detection completed: {len(anomalies)} anomalies found")
                for anomaly in anomalies[:3]:  # Show first 3
                    logger.info(f"   - [{anomaly['severity']}] {anomaly['type']}: {anomaly['message']}")
            except Exception as e:
                logger.error(f"❌ Anomaly detection failed: {e}")
            
            # Test 5: Performance metrics
            logger.info("\n5. Testing performance metrics...")
            try:
                performance = await analytics.get_performance_metrics(hours=24)
                logger.info("✅ Performance metrics retrieved successfully")
                logger.info(f"   Average latency: {performance['overall']['avg_latency']:.0f}ms")
                logger.info(f"   P95 latency: {performance['overall']['p95_latency']:.0f}ms")
                logger.info(f"   P99 latency: {performance['overall']['p99_latency']:.0f}ms")
                logger.info(f"   Total requests: {performance['overall']['total_requests']}")
            except Exception as e:
                logger.error(f"❌ Performance metrics failed: {e}")
            
            # Test 6: Generate sample report
            logger.info("\n6. Testing report generation...")
            try:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=7)
                report = await analytics.generate_usage_report(start_date, end_date)
                
                logger.info("✅ Report generated successfully")
                logger.info(f"   Report period: {report['report_metadata']['period']['days']} days")
                logger.info(f"   Total requests: {report['executive_summary']['total_requests']}")
                logger.info(f"   Total cost: ${report['executive_summary']['total_cost']:.2f}")
                logger.info(f"   Anomalies: {report['executive_summary']['anomalies_detected']}")
                logger.info(f"   Recommendations: {len(report['recommendations'])}")
                
                # Save sample report
                report_path = Path("reports") / "sample_report.json"
                report_path.parent.mkdir(exist_ok=True)
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"   Sample report saved to: {report_path}")
                
            except Exception as e:
                logger.error(f"❌ Report generation failed: {e}")
            
            # Test 7: User-specific trends
            logger.info("\n7. Testing user-specific trends...")
            try:
                # Get a sample user ID
                result = await session.execute(
                    "SELECT TOP 1 user_id FROM reg_llm_requests ORDER BY created_at DESC"
                )
                user_row = result.fetchone()
                
                if user_row:
                    user_id = user_row.user_id
                    user_trends = await analytics.get_usage_trends(days=7, user_id=user_id)
                    logger.info(f"✅ User trends retrieved for user_id: {user_id}")
                    logger.info(f"   User requests: {user_trends['summary']['total_requests']}")
                    logger.info(f"   User cost: ${user_trends['summary']['total_cost']:.2f}")
                else:
                    logger.info("⚠️  No user data available for user-specific trends")
                    
            except Exception as e:
                logger.error(f"❌ User-specific trends failed: {e}")


async def test_monitoring_endpoints():
    """Test monitoring API endpoints"""
    logger.info("\n\nTesting Monitoring API Endpoints...")
    
    import httpx
    
    BASE_URL = "http://localhost:8000/api/v1"
    
    # First, get auth token
    async with httpx.AsyncClient() as client:
        # Login as admin
        try:
            response = await client.post(
                f"{BASE_URL}/auth/login",
                data={
                    "username": "admin@datafactz.com",
                    "password": "password123"
                }
            )
            if response.status_code == 200:
                token = response.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                logger.info("✅ Admin authentication successful")
            else:
                logger.error("❌ Failed to authenticate")
                return
        except Exception as e:
            logger.error(f"❌ Auth error: {e}")
            return
        
        # Test dashboard overview
        try:
            response = await client.get(
                f"{BASE_URL}/monitoring/dashboard/overview?days=7",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Dashboard overview retrieved")
                logger.info(f"   Total requests: {data['summary']['total_requests']}")
                logger.info(f"   Anomalies: {data['summary']['anomaly_count']}")
            else:
                logger.error(f"❌ Dashboard overview failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Dashboard error: {e}")
        
        # Test anomaly detection endpoint
        try:
            response = await client.get(
                f"{BASE_URL}/monitoring/analytics/anomalies",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Anomaly detection endpoint working")
                logger.info(f"   Total anomalies: {data['total_count']}")
                logger.info(f"   Requires action: {data['requires_action']}")
            else:
                logger.error(f"❌ Anomaly endpoint failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Anomaly endpoint error: {e}")


async def main():
    """Run all analytics tests"""
    logger.info("Starting Analytics Service tests...")
    
    # Test analytics service
    await test_analytics_service()
    
    # Test monitoring endpoints (only if API is running)
    logger.info("\n" + "="*60)
    logger.info("To test monitoring endpoints, ensure the API is running:")
    logger.info("python -m uvicorn src.main:app --reload")
    
    # Uncomment to test endpoints
    # await test_monitoring_endpoints()
    
    logger.info("\n✅ Analytics tests completed!")


if __name__ == "__main__":
    asyncio.run(main())