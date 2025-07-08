"""
Application configuration settings using Pydantic BaseSettings
"""
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application Settings
    app_name: str = Field(default="regulatory-query-agent", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_env: str = Field(default="development", description="Application environment")
    app_port: int = Field(default=8000, description="Application port")
    app_host: str = Field(default="0.0.0.0", description="Application host")
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API route prefix")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed CORS origins"
    )
    
    # Azure AI Search Configuration
    azure_search_endpoint: str = Field(..., description="Azure AI Search endpoint")
    azure_search_key: str = Field(..., description="Azure AI Search API key")
    azure_search_index_name: str = Field(..., description="Azure AI Search index name")
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(..., description="Azure OpenAI endpoint")
    azure_openai_api_key: str = Field(..., description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(..., description="Azure OpenAI API version")
    azure_openai_embedding_deployment: str = Field(..., description="Embedding model deployment name")
    azure_openai_deployment_name: str = Field(..., description="Chat model deployment name")
    azure_openai_default_model: str = Field(default="gpt-4", description="Default chat model")
    
    # Model-specific deployments for different tasks
    azure_openai_gpt4_deployment: str = Field(default="gpt-4", description="GPT-4 deployment name")
    azure_openai_gpt4o_deployment: str = Field(default="gpt-4o", description="GPT-4o deployment name")
    azure_openai_gpt35_deployment: str = Field(default="gpt-35-turbo", description="GPT-3.5 deployment name")
    
    # Task-specific model configurations
    azure_openai_intent_model: str = Field(default="gpt-4", description="Model for intent analysis")
    azure_openai_concept_model: str = Field(default="gpt-4", description="Model for concept expansion")
    azure_openai_summary_model: str = Field(default="gpt-35-turbo", description="Model for summarization")
    azure_openai_extract_model: str = Field(default="gpt-35-turbo", description="Model for extraction")
    
    
    # Azure SQL Configuration
    sql_use_azure: bool = Field(default=True, description="Use Azure SQL")
    sql_server: str = Field(..., description="SQL Server hostname")
    sql_database: str = Field(..., description="SQL Database name")
    sql_username: str = Field(..., description="SQL Username")
    sql_password: str = Field(..., description="SQL Password")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str = Field(default="", description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # Intent Cache Configuration
    intent_cache_enabled: bool = Field(default=False, description="Enable intent-based caching")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json or plain)")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, description="API rate limit per minute")
    
    # Session Configuration
    session_timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
    max_conversation_history: int = Field(default=20, description="Maximum conversation history")
    
    # LLM Configuration
    llm_temperature: float = Field(default=0.1, description="LLM temperature for response generation")
    llm_max_tokens: int = Field(default=3000, description="Maximum tokens for LLM response")
    llm_top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    
    # Search Configuration
    search_top_k: int = Field(default=30, description="Number of top results to retrieve")
    search_min_score: float = Field(default=0.3, description="Minimum relevance score")
    hybrid_search_weight: float = Field(default=0.5, description="Weight for hybrid search (0=keyword, 1=vector)")
    
    # JWT Configuration
    jwt_secret_key: str = Field(default="your-secret-key-please-change", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(default=60, description="JWT token expiration in minutes")
    
    @validator("app_env")
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"app_env must be one of {allowed_envs}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"log_level must be one of {allowed_levels}")
        return v.upper()
    
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.app_env == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.app_env == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create a global settings instance
settings = get_settings()