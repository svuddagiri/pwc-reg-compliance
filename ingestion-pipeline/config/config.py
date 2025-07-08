from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_title: str = "Regulatory Compliance Chatbot API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    # Authentication
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_deployment_name: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    
    # Azure Search
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index_name: str = "regulatory-hybrid-semantic-name-test"
    
    # Azure Storage
    azure_storage_connection_string: str
    azure_storage_container_name: str = "databreach"
    
    # Azure Document Intelligence
    azure_document_intelligence_endpoint: str
    azure_document_intelligence_key: str
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    # # Azure Cosmos DB (Optional - for backward compatibility)
    # cosmos_endpoint: Optional[str] = None
    # cosmos_key: Optional[str] = None
    # cosmos_database_name: str = "regulatory-hybrid-semantic-name-test"
    # cosmos_users_container: str = "users"
    # cosmos_audit_container: str = "audit-trail"
    
    # Performance Settings
    max_response_time_seconds: float = 5.0
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    max_documents_mvp: int = 50
    
    # Retrieval Settings
    top_k_retrieval: int = 5
    hybrid_search_weight: float = 0.5  # Balance between vector and keyword search
    
    # SQL Server Configuration
    sql_use_azure: bool = False  # Set to True for Azure SQL Database
    sql_server: str = "localhost"
    sql_database: str = "regulatory_compliance"
    sql_username: str = "sa"
    sql_password: str = "YourStrong@Passw0rd"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env file

settings = Settings()