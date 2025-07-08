"""SQL Server database implementation using SQLAlchemy"""

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
import uuid
from datetime import datetime
from config.config import settings
import structlog

logger = structlog.get_logger()

Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    password_changed_at = Column(DateTime, nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)  # Changed from username
    action = Column(String(20), nullable=False, index=True)
    resource_type = Column(String(10), nullable=False)
    resource_id = Column(String(255), nullable=False)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    
    __table_args__ = (
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
    )

class PipelineJob(Base):
    __tablename__ = "pipeline_jobs"
    
    job_id = Column(String(36), primary_key=True)
    created_by = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    status = Column(String(20), nullable=False, index=True)
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    failed_documents = Column(Integer, default=0)
    documents = Column(JSON)  # List of document names
    error_details = Column(JSON)
    completed_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_job_status_date', 'status', 'created_at'),
        Index('idx_job_user_date', 'created_by', 'created_at'),
    )

class DocumentStatus(Base):
    __tablename__ = "document_status"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), unique=True, nullable=False, index=True)
    document_name = Column(String(255), nullable=False, index=True)
    blob_name = Column(String(255), nullable=False, unique=True)
    document_type = Column(String(50))
    total_pages = Column(Integer)
    total_sections = Column(Integer)
    total_chunks = Column(Integer)
    status = Column(String(20), nullable=False, index=True)
    processed_at = Column(DateTime, default=func.now())
    processing_time_seconds = Column(Integer)
    error_message = Column(Text)
    doc_metadata = Column(JSON)  # Store extracted entities, penalties, etc.
    file_hash = Column(String(64))  # SHA256 hash of file content
    file_size = Column(Integer)  # File size in bytes
    last_modified = Column(DateTime)  # Last modified time from blob storage
    
    __table_args__ = (
        Index('idx_doc_name_status', 'document_name', 'status'),
        Index('idx_doc_processed_date', 'processed_at'),
    )

class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    sources = Column(JSON)  # List of source documents
    timestamp = Column(DateTime, default=func.now(), index=True)
    response_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    
    __table_args__ = (
        Index('idx_query_user_date', 'username', 'timestamp'),
    )

# Database Manager
class SQLDatabase:
    def __init__(self):
        # Use DATABASE_URL from environment if available
        import os
        if os.getenv('DATABASE_URL'):
            connection_string = os.getenv('DATABASE_URL')
        elif settings.sql_use_azure:
            # Azure SQL Database connection string
            connection_string = (
                f"mssql+pyodbc://{settings.sql_username}:{settings.sql_password}@"
                f"{settings.sql_server}/{settings.sql_database}"
                f"?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
            )
        else:
            # Local SQL Server connection string
            connection_string = (
                f"mssql+pyodbc://{settings.sql_username}:{settings.sql_password}@"
                f"{settings.sql_server}/{settings.sql_database}"
                f"?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
            )
        
        self.engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
        # Don't create tables on initialization - let them be created manually
        logger.info("SQL Database initialized (tables not auto-created)")
    
    def _create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error("Failed to create tables", error=str(e))
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Provide a transactional scope for database operations"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # User Management Methods
    def create_user(self, username: str, email: str, hashed_password: str, 
                   is_admin: bool = False) -> Dict[str, Any]:
        """Create a new user"""
        with self.get_session() as session:
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                is_superuser=is_admin
            )
            session.add(user)
            session.flush()
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "is_admin": user.is_superuser,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat()
            }
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return None
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "hashed_password": user.hashed_password,
                "is_admin": user.is_superuser,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat()
            }
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                return False
            
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            user.updated_at = datetime.utcnow()
            return True
    
    # Audit Log Methods
    def create_audit_log(self, username: str, action: str, resource: Optional[str] = None,
                        details: Optional[Dict] = None, ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None, status: str = "success",
                        error_message: Optional[str] = None):
        """Create an audit log entry"""
        with self.get_session() as session:
            # Parse resource into type and id
            resource_type = "general"
            resource_id = resource or "system"
            if resource and "/" in resource:
                parts = resource.split("/", 1)
                resource_type = parts[0][:10]  # Max 10 chars for resource_type
                resource_id = parts[1]
            
            # Get user_id from username
            user = session.query(User).filter(User.username == username).first()
            user_id = user.id if user else username
            
            audit = AuditLog(
                user_id=user_id,
                action=action[:20],  # Max 20 chars for action
                resource_type=resource_type,
                resource_id=resource_id,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            session.add(audit)
    
    def get_audit_logs(self, username: Optional[str] = None, action: Optional[str] = None,
                      start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        with self.get_session() as session:
            query = session.query(AuditLog, User).join(
                User, AuditLog.user_id == User.id, isouter=True
            )
            
            if username:
                query = query.filter(User.username == username)
            if action:
                query = query.filter(AuditLog.action == action)
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)
            
            results = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
            
            return [{
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "username": user.username if user else log.user_id,
                "action": log.action,
                "resource": f"{log.resource_type}/{log.resource_id}",
                "details": log.details,
                "status": "success",  # Not stored in new schema
                "error_message": None  # Not stored in new schema
            } for log, user in results]
    
    # Pipeline Job Methods
    def create_pipeline_job(self, job_id: str, created_by: str, documents: List[str]) -> Dict[str, Any]:
        """Create a new pipeline job"""
        with self.get_session() as session:
            job = PipelineJob(
                job_id=job_id,
                created_by=created_by,
                status="initializing",
                total_documents=len(documents),
                documents=documents
            )
            session.add(job)
            return {
                "job_id": job.job_id,
                "created_by": job.created_by,
                "status": job.status,
                "total_documents": job.total_documents
            }
    
    def update_pipeline_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update pipeline job status"""
        with self.get_session() as session:
            job = session.query(PipelineJob).filter(PipelineJob.job_id == job_id).first()
            if not job:
                return False
            
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            job.updated_at = datetime.utcnow()
            return True
    
    def get_pipeline_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline job by ID"""
        with self.get_session() as session:
            job = session.query(PipelineJob).filter(PipelineJob.job_id == job_id).first()
            if not job:
                return None
            
            return {
                "job_id": job.job_id,
                "created_by": job.created_by,
                "created_at": job.created_at.isoformat(),
                "status": job.status,
                "total_documents": job.total_documents,
                "processed_documents": job.processed_documents,
                "failed_documents": job.failed_documents,
                "documents": job.documents,
                "error_details": job.error_details
            }
    
    def get_recent_pipeline_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent pipeline jobs"""
        with self.get_session() as session:
            jobs = session.query(PipelineJob)\
                         .order_by(PipelineJob.created_at.desc())\
                         .limit(limit)\
                         .all()
            
            return [{
                "job_id": job.job_id,
                "created_by": job.created_by,
                "created_at": job.created_at.isoformat(),
                "status": job.status,
                "total_documents": job.total_documents,
                "processed_documents": job.processed_documents
            } for job in jobs]
    
    # Document Status Methods
    def upsert_document_status(self, document_data: Dict[str, Any]) -> bool:
        """Insert or update document status"""
        with self.get_session() as session:
            # Check if document exists by blob_name (which has unique constraint)
            doc = session.query(DocumentStatus).filter(
                DocumentStatus.blob_name == document_data.get("blob_name")
            ).first()
            
            if doc:
                # Update existing
                for key, value in document_data.items():
                    if hasattr(doc, key) and key != 'id':  # Don't update primary key
                        setattr(doc, key, value)
            else:
                # Create new with new ID
                document_data['id'] = str(uuid.uuid4())
                doc = DocumentStatus(**document_data)
                session.add(doc)
            
            return True
    
    def is_document_processed(self, blob_name: str) -> bool:
        """Check if a document has been processed"""
        with self.get_session() as session:
            doc = session.query(DocumentStatus).filter(
                DocumentStatus.blob_name == blob_name,
                DocumentStatus.status == "completed"
            ).first()
            return doc is not None
    
    def check_document_changed(self, blob_name: str, file_hash: str, file_size: int, 
                              last_modified: datetime) -> Tuple[bool, Optional[str]]:
        """Check if document content or metadata has changed
        Returns: (needs_processing, reason)
        """
        with self.get_session() as session:
            doc = session.query(DocumentStatus).filter(
                DocumentStatus.blob_name == blob_name
            ).first()
            
            if not doc:
                return True, "new_document"
            
            if doc.status != "completed":
                return True, f"incomplete_status:{doc.status}"
            
            # Check for content changes
            if doc.file_hash and doc.file_hash != file_hash:
                return True, "content_changed"
            
            # Check for size changes (fallback if no hash)
            if doc.file_size and doc.file_size != file_size:
                return True, "size_changed"
            
            # Check for modification time (if available)
            if doc.last_modified and last_modified and doc.last_modified < last_modified:
                return True, "modified_recently"
            
            return False, None
    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get list of successfully processed documents"""
        with self.get_session() as session:
            docs = session.query(DocumentStatus).filter(
                DocumentStatus.status == "completed"
            ).all()
            
            return [{"blob_name": doc.blob_name} for doc in docs]
    
    def get_failed_documents(self) -> List[Dict[str, Any]]:
        """Get list of failed documents"""
        with self.get_session() as session:
            docs = session.query(DocumentStatus).filter(
                DocumentStatus.status == "failed"
            ).all()
            
            return [{"blob_name": doc.blob_name} for doc in docs]
    
    def store_document_metadata(self, metadata: Dict[str, Any]):
        """Store document metadata after processing"""
        with self.get_session() as session:
            # Update document status with metadata
            doc = session.query(DocumentStatus).filter(
                DocumentStatus.document_id == metadata.get("document_id")
            ).first()
            
            if doc:
                doc.doc_metadata = metadata.get("metadata_json", {})
                doc.total_chunks = metadata.get("total_chunks", 0)
                doc.processed_at = datetime.utcnow()
    
    def get_document_details(self, document_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed document information"""
        with self.get_session() as session:
            doc = session.query(DocumentStatus).filter(
                DocumentStatus.document_name == document_name
            ).first()
            
            if not doc:
                return None
            
            return {
                "document_id": doc.document_id,
                "document_name": doc.document_name,
                "status": doc.status,
                "document_type": doc.document_type,
                "regulatory_framework": doc.doc_metadata.get("regulatory_framework", "Unknown"),
                "jurisdiction": doc.doc_metadata.get("jurisdiction", "Unknown"),
                "total_chunks": doc.total_chunks,
                "total_entities": doc.doc_metadata.get("total_entities", 0),
                "total_relationships": doc.doc_metadata.get("total_relationships", 0),
                "validation_issues": doc.doc_metadata.get("validation_issues", []),
                "processing_time": doc.doc_metadata.get("processing_time", 0.0),
                "quality_scores": doc.doc_metadata.get("quality_scores", {})
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get overall processing statistics"""
        with self.get_session() as session:
            # Get job statistics
            total_jobs = session.query(PipelineJob).count()
            completed_jobs = session.query(PipelineJob).filter(
                PipelineJob.status == "completed"
            ).count()
            failed_jobs = session.query(PipelineJob).filter(
                PipelineJob.status == "failed"
            ).count()
            
            # Get document statistics
            total_docs = session.query(DocumentStatus).filter(
                DocumentStatus.status == "completed"
            ).count()
            
            # Get average processing time from completed jobs
            # Calculate duration from created_at and completed_at
            completed_jobs_with_time = session.query(PipelineJob)\
                .filter(PipelineJob.status == "completed")\
                .filter(PipelineJob.completed_at.isnot(None))\
                .all()
            
            if completed_jobs_with_time:
                durations = [
                    (job.completed_at - job.created_at).total_seconds()
                    for job in completed_jobs_with_time
                ]
                avg_time = sum(durations) / len(durations)
            else:
                avg_time = 0.0
            
            # Get total chunks and entities from metadata
            chunks_query = session.query(func.sum(DocumentStatus.total_chunks))\
                                .filter(DocumentStatus.status == "completed")\
                                .scalar() or 0
            
            # Get document type distribution
            doc_types = session.query(
                DocumentStatus.document_type,
                func.count(DocumentStatus.id)
            ).filter(
                DocumentStatus.status == "completed"
            ).group_by(
                DocumentStatus.document_type
            ).all()
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "total_documents_processed": total_docs,
                "average_processing_time": float(avg_time),
                "total_chunks_created": int(chunks_query),
                "total_entities_extracted": 0,  # Would need to aggregate from metadata
                "document_types": dict(doc_types) if doc_types else {},
                "regulatory_frameworks": {}  # Would need to extract from metadata
            }
    
    def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status"""
        with self.get_session() as session:
            doc = session.query(DocumentStatus).filter(
                DocumentStatus.document_id == document_id
            ).first()
            
            if not doc:
                return None
            
            return {
                "document_id": doc.document_id,
                "document_name": doc.document_name,
                "status": doc.status,
                "total_chunks": doc.total_chunks,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "error_message": doc.error_message
            }
    
    # Query History Methods
    def save_query(self, username: str, question: str, answer: str, 
                  sources: List[Dict], response_time_ms: int, tokens_used: int):
        """Save a Q&A query to history"""
        with self.get_session() as session:
            query = QueryHistory(
                username=username,
                question=question,
                answer=answer,
                sources=sources,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used
            )
            session.add(query)
    
    def get_query_history(self, username: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get query history"""
        with self.get_session() as session:
            query = session.query(QueryHistory)
            
            if username:
                query = query.filter(QueryHistory.username == username)
            
            queries = query.order_by(QueryHistory.timestamp.desc()).limit(limit).all()
            
            return [{
                "id": q.id,
                "username": q.username,
                "question": q.question,
                "answer": q.answer,
                "sources": q.sources,
                "timestamp": q.timestamp.isoformat(),
                "response_time_ms": q.response_time_ms
            } for q in queries]

# Create global instance lazily
class LazySQL:
    _instance = None
    
    def __getattr__(self, name):
        if self._instance is None:
            self._instance = SQLDatabase()
        return getattr(self._instance, name)

sql_db = LazySQL()