from azure.cosmos import CosmosClient, PartitionKey, exceptions
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
from config.config import settings
import structlog

logger = structlog.get_logger()

class CosmosDBClient:
    def __init__(self):
        self.client = CosmosClient(settings.cosmos_endpoint, settings.cosmos_key)
        self.database = None
        self.users_container = None
        self.audit_container = None
        self.pipeline_jobs_container = None
        self.document_status_container = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize Cosmos DB database and containers"""
        try:
            # Create database if it doesn't exist
            self.database = self.client.create_database_if_not_exists(
                id=settings.cosmos_database_name
            )
            
            # Create users container with username as partition key
            self.users_container = self.database.create_container_if_not_exists(
                id=settings.cosmos_users_container,
                partition_key=PartitionKey(path="/username")
                # No throughput for serverless
            )
            
            # Create audit trail container with date as partition key for efficient querying
            self.audit_container = self.database.create_container_if_not_exists(
                id=settings.cosmos_audit_container,
                partition_key=PartitionKey(path="/date")
                # No throughput for serverless
            )
            
            # Create pipeline jobs container for tracking document processing
            self.pipeline_jobs_container = self.database.create_container_if_not_exists(
                id="pipeline-jobs",
                partition_key=PartitionKey(path="/job_id")
                # No throughput for serverless
            )
            
            # Create document status container for tracking individual document processing
            self.document_status_container = self.database.create_container_if_not_exists(
                id="document-status",
                partition_key=PartitionKey(path="/document_name")
                # No throughput for serverless
            )
            
            logger.info("Cosmos DB initialized successfully")
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error("Failed to initialize Cosmos DB", error=str(e))
            raise
    
    def create_user(self, user_data: Dict) -> Dict:
        """Create a new user in Cosmos DB"""
        try:
            user_data["id"] = str(uuid.uuid4())
            user_data["created_at"] = datetime.utcnow().isoformat()
            user_data["type"] = "user"
            
            created_user = self.users_container.create_item(body=user_data)
            logger.info("User created in Cosmos DB", username=user_data.get("username"))
            return created_user
            
        except exceptions.CosmosResourceExistsError:
            raise ValueError("User already exists")
        except Exception as e:
            logger.error("Failed to create user", error=str(e))
            raise
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            query = "SELECT * FROM c WHERE c.username = @username AND c.type = 'user'"
            parameters = [{"name": "@username", "value": username}]
            
            items = list(self.users_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return items[0] if items else None
            
        except Exception as e:
            logger.error("Failed to get user", error=str(e), username=username)
            return None
    
    def update_user(self, username: str, updates: Dict) -> bool:
        """Update user data"""
        try:
            user = self.get_user(username)
            if not user:
                return False
            
            user.update(updates)
            user["updated_at"] = datetime.utcnow().isoformat()
            
            self.users_container.upsert_item(body=user)
            logger.info("User updated", username=username)
            return True
            
        except Exception as e:
            logger.error("Failed to update user", error=str(e), username=username)
            return False
    
    def delete_user(self, username: str) -> bool:
        """Delete user"""
        try:
            user = self.get_user(username)
            if not user:
                return False
            
            self.users_container.delete_item(
                item=user["id"],
                partition_key=username
            )
            logger.info("User deleted", username=username)
            return True
            
        except Exception as e:
            logger.error("Failed to delete user", error=str(e), username=username)
            return False
    
    def list_users(self) -> List[Dict]:
        """List all users"""
        try:
            query = "SELECT * FROM c WHERE c.type = 'user'"
            items = list(self.users_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items
            
        except Exception as e:
            logger.error("Failed to list users", error=str(e))
            return []
    
    def add_audit_entry(self, entry_data: Dict) -> Dict:
        """Add an entry to the audit trail"""
        try:
            entry_data["id"] = str(uuid.uuid4())
            entry_data["timestamp"] = datetime.utcnow().isoformat()
            entry_data["date"] = datetime.utcnow().strftime("%Y-%m-%d")  # Partition key
            entry_data["type"] = "audit"
            
            created_entry = self.audit_container.create_item(body=entry_data)
            return created_entry
            
        except Exception as e:
            logger.error("Failed to add audit entry", error=str(e))
            raise
    
    def get_audit_trail(self, start_date: str = None, end_date: str = None, 
                            user: str = None) -> List[Dict]:
        """Get audit trail entries with optional filters"""
        try:
            query = "SELECT * FROM c WHERE c.type = 'audit'"
            parameters = []
            
            if start_date:
                query += " AND c.date >= @start_date"
                parameters.append({"name": "@start_date", "value": start_date})
            
            if end_date:
                query += " AND c.date <= @end_date"
                parameters.append({"name": "@end_date", "value": end_date})
            
            if user:
                query += " AND c.username = @username"
                parameters.append({"name": "@username", "value": user})
            
            query += " ORDER BY c.timestamp DESC"
            
            items = list(self.audit_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return items
            
        except Exception as e:
            logger.error("Failed to get audit trail", error=str(e))
            return []

# Global instance
cosmos_db = CosmosDBClient()