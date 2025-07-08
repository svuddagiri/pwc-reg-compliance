"""
Daily analytics aggregation job

This script should be run daily via cron job or task scheduler
to aggregate LLM usage statistics and generate reports.
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.analytics_service import AnalyticsService
from src.clients.azure_sql import AzureSQLClient
from src.utils.logger import get_logger
from src.config import settings

logger = get_logger(__name__)


class DailyAnalyticsJob:
    """Daily analytics aggregation and reporting job"""
    
    def __init__(self):
        self.analytics_service = None
        self.sql_client = None
    
    async def run(self, target_date=None):
        """Run the daily analytics job"""
        try:
            logger.info("Starting daily analytics job")
            
            # Initialize services
            self.sql_client = AzureSQLClient()
            async with self.sql_client.get_session() as session:
                self.analytics_service = AnalyticsService(session)
                
                # 1. Aggregate yesterday's data
                if not target_date:
                    target_date = (datetime.utcnow() - timedelta(days=1)).date()
                
                logger.info(f"Aggregating data for {target_date}")
                daily_stats = await self.analytics_service.aggregate_daily_usage(target_date)
                
                # 2. Detect anomalies
                logger.info("Detecting anomalies...")
                anomalies = await self.analytics_service.detect_anomalies()
                
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalies")
                    for anomaly in anomalies:
                        logger.warning(f"- {anomaly['message']}")
                
                # 3. Generate weekly report (on Mondays)
                if datetime.utcnow().weekday() == 0:  # Monday
                    logger.info("Generating weekly report...")
                    await self._generate_weekly_report()
                
                # 4. Generate monthly report (on the 1st)
                if datetime.utcnow().day == 1:
                    logger.info("Generating monthly report...")
                    await self._generate_monthly_report()
                
                # 5. Check for critical alerts
                await self._check_critical_alerts(anomalies)
                
                # 6. Cleanup old data (keep last 90 days of detailed logs)
                await self._cleanup_old_data()
                
                logger.info("Daily analytics job completed successfully")
                
        except Exception as e:
            logger.error(f"Daily analytics job failed: {e}", exc_info=True)
            await self._send_error_notification(str(e))
            raise
    
    async def _generate_weekly_report(self):
        """Generate and send weekly usage report"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            report = await self.analytics_service.generate_usage_report(
                start_date, end_date, include_details=True
            )
            
            # Save report to file
            report_path = Path("reports") / f"weekly_report_{end_date.strftime('%Y%m%d')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Weekly report saved to {report_path}")
            
            # Send email if configured
            if settings.email_notifications_enabled:
                await self._send_report_email(report, "Weekly")
                
        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}", exc_info=True)
    
    async def _generate_monthly_report(self):
        """Generate and send monthly usage report"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date.replace(day=1) - timedelta(days=1)
            start_date = start_date.replace(day=1)
            
            report = await self.analytics_service.generate_usage_report(
                start_date, end_date, include_details=True
            )
            
            # Add month-specific analysis
            report["monthly_analysis"] = {
                "cost_trend": self._analyze_cost_trend(report),
                "user_growth": self._analyze_user_growth(report),
                "performance_summary": self._analyze_performance(report)
            }
            
            # Save report
            report_path = Path("reports") / f"monthly_report_{end_date.strftime('%Y%m')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Monthly report saved to {report_path}")
            
            # Send email if configured
            if settings.email_notifications_enabled:
                await self._send_report_email(report, "Monthly")
                
        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}", exc_info=True)
    
    async def _check_critical_alerts(self, anomalies):
        """Check for critical conditions that need immediate attention"""
        critical_conditions = []
        
        # High error rate
        high_error_anomalies = [
            a for a in anomalies 
            if a["type"] == "high_error_rate" and a["severity"] == "high"
        ]
        if high_error_anomalies:
            critical_conditions.append({
                "type": "high_error_rate",
                "message": f"System experiencing high error rates: {high_error_anomalies[0]['details']['error_rate']}%",
                "action": "Investigate LLM service health immediately"
            })
        
        # Security threats
        security_anomalies = [
            a for a in anomalies 
            if a["type"] == "security_event" and a["severity"] in ["high", "critical"]
        ]
        if len(security_anomalies) > 3:
            critical_conditions.append({
                "type": "security_threat",
                "message": f"Multiple high-severity security events detected: {len(security_anomalies)} events",
                "action": "Review security logs and potentially block affected users"
            })
        
        # Cost spike
        async with self.sql_client.get_session() as session:
            analytics = AnalyticsService(session)
            cost_data = await analytics.get_cost_breakdown()
            
            daily_cost = cost_data["cost_projection"]["daily"]
            if daily_cost > settings.daily_cost_alert_threshold:
                critical_conditions.append({
                    "type": "cost_spike",
                    "message": f"Daily cost (${daily_cost:.2f}) exceeds threshold (${settings.daily_cost_alert_threshold})",
                    "action": "Review usage patterns and consider implementing stricter rate limits"
                })
        
        # Send alerts if any critical conditions
        if critical_conditions:
            await self._send_critical_alert(critical_conditions)
    
    async def _cleanup_old_data(self):
        """Clean up old detailed logs while preserving aggregated data"""
        try:
            retention_date = datetime.utcnow() - timedelta(days=90)
            
            async with self.sql_client.get_session() as session:
                # Delete old LLM responses (keep request metadata)
                result = await session.execute(
                    """
                    DELETE FROM reg_llm_responses 
                    WHERE created_at < ? 
                    AND request_id IN (
                        SELECT request_id FROM reg_llm_requests 
                        WHERE created_at < ?
                    )
                    """,
                    (retention_date, retention_date)
                )
                
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old LLM response records")
                
                # Delete old security events (low severity only)
                result = await session.execute(
                    """
                    DELETE FROM reg_security_events 
                    WHERE created_at < ? 
                    AND severity = 'low'
                    """,
                    (retention_date,)
                )
                
                if result.rowcount > 0:
                    logger.info(f"Cleaned up {result.rowcount} old security events")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}", exc_info=True)
    
    async def _send_report_email(self, report, report_type):
        """Send report via email"""
        try:
            # Format email content
            subject = f"[Regulatory Query Agent] {report_type} Analytics Report"
            
            html_content = f"""
            <html>
            <body>
                <h2>{report_type} Analytics Report</h2>
                <p>Report Period: {report['report_metadata']['period']['start']} to {report['report_metadata']['period']['end']}</p>
                
                <h3>Executive Summary</h3>
                <ul>
                    <li>Total Requests: {report['executive_summary']['total_requests']:,}</li>
                    <li>Total Cost: ${report['executive_summary']['total_cost']:.2f}</li>
                    <li>Average Daily Cost: ${report['executive_summary']['avg_daily_cost']:.2f}</li>
                    <li>Unique Users: {report['executive_summary']['unique_users']}</li>
                    <li>Avg Response Time: {report['executive_summary']['avg_response_time']:.0f}ms</li>
                    <li>Anomalies Detected: {report['executive_summary']['anomalies_detected']}</li>
                </ul>
                
                <h3>Key Recommendations</h3>
                <ol>
                    {"".join(f"<li><b>{r['category']}:</b> {r['recommendation']}</li>" for r in report['recommendations'][:3])}
                </ol>
                
                <p>Full report available in the admin dashboard.</p>
            </body>
            </html>
            """
            
            await self._send_email(
                settings.admin_email_addresses,
                subject,
                html_content
            )
            
        except Exception as e:
            logger.error(f"Failed to send report email: {e}", exc_info=True)
    
    async def _send_critical_alert(self, conditions):
        """Send critical alert notifications"""
        try:
            subject = "[URGENT] Regulatory Query Agent - Critical Alert"
            
            html_content = """
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">Critical System Alert</h2>
                <p>The following critical conditions have been detected:</p>
            """
            
            for condition in conditions:
                html_content += f"""
                <div style="border: 1px solid #d32f2f; padding: 10px; margin: 10px 0;">
                    <h3 style="color: #d32f2f;">{condition['type'].replace('_', ' ').title()}</h3>
                    <p><b>Issue:</b> {condition['message']}</p>
                    <p><b>Recommended Action:</b> {condition['action']}</p>
                </div>
                """
            
            html_content += """
                <p>Please investigate immediately.</p>
                <p><small>This is an automated alert from the Regulatory Query Agent analytics system.</small></p>
            </body>
            </html>
            """
            
            await self._send_email(
                settings.admin_email_addresses,
                subject,
                html_content,
                priority="high"
            )
            
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}", exc_info=True)
    
    async def _send_error_notification(self, error_message):
        """Send error notification for job failure"""
        try:
            subject = "[ERROR] Regulatory Query Agent - Analytics Job Failed"
            
            html_content = f"""
            <html>
            <body>
                <h2>Analytics Job Failure</h2>
                <p>The daily analytics job has failed with the following error:</p>
                <pre style="background: #f5f5f5; padding: 10px; border: 1px solid #ddd;">
{error_message}
                </pre>
                <p>Please check the logs for more details.</p>
            </body>
            </html>
            """
            
            await self._send_email(
                settings.admin_email_addresses,
                subject,
                html_content,
                priority="high"
            )
            
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}", exc_info=True)
    
    async def _send_email(self, recipients, subject, html_content, priority="normal"):
        """Send email helper (placeholder - implement based on your email service)"""
        # This is a placeholder - implement based on your email service
        # Options: SendGrid, AWS SES, SMTP, etc.
        logger.info(f"Would send email to {recipients}: {subject}")
    
    def _analyze_cost_trend(self, report):
        """Analyze cost trends from report data"""
        trends = report.get("usage_trends", {}).get("trends", {})
        cost_trend = trends.get("cost", 0)
        
        if cost_trend > 20:
            return {"status": "increasing", "percentage": cost_trend, "alert": True}
        elif cost_trend < -20:
            return {"status": "decreasing", "percentage": cost_trend, "alert": False}
        else:
            return {"status": "stable", "percentage": cost_trend, "alert": False}
    
    def _analyze_user_growth(self, report):
        """Analyze user growth from report data"""
        daily_data = report.get("usage_trends", {}).get("daily_data", [])
        if len(daily_data) < 2:
            return {"new_users": 0, "growth_rate": 0}
        
        # Simple approximation - would need more detailed data in production
        first_week_users = daily_data[0].get("unique_users", 0) if daily_data else 0
        last_week_users = daily_data[-1].get("unique_users", 0) if daily_data else 0
        
        growth = ((last_week_users - first_week_users) / first_week_users * 100) if first_week_users > 0 else 0
        
        return {
            "total_users": report.get("executive_summary", {}).get("unique_users", 0),
            "growth_rate": round(growth, 1)
        }
    
    def _analyze_performance(self, report):
        """Analyze performance trends from report data"""
        perf = report.get("performance_metrics", {}).get("overall", {})
        
        return {
            "avg_latency": perf.get("avg_latency", 0),
            "p95_latency": perf.get("p95_latency", 0),
            "p99_latency": perf.get("p99_latency", 0),
            "status": "healthy" if perf.get("p95_latency", 0) < 5000 else "degraded"
        }


async def main():
    """Run the daily analytics job"""
    job = DailyAnalyticsJob()
    await job.run()


if __name__ == "__main__":
    asyncio.run(main())