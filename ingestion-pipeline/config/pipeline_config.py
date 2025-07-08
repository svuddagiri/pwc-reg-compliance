"""
Modular Pipeline Configuration
Allows easy configuration of pipeline components and behavior
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import json
from pathlib import Path


@dataclass
class PipelineConfig:
    """Pipeline configuration settings"""
    
    # Processing settings
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: int = 5
    
    # Document processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    
    # Storage settings
    enable_cosmos_db: bool = True
    enable_azure_search: bool = True
    enable_sql_db: bool = True
    enable_redis_cache: bool = False
    
    # AI enrichment settings
    enable_embeddings: bool = True
    enable_ner: bool = True
    enable_metadata_extraction: bool = True
    enable_cross_reference_detection: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from file (JSON or YAML)"""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_chunk_size': self.min_chunk_size,
            'enable_cosmos_db': self.enable_cosmos_db,
            'enable_azure_search': self.enable_azure_search,
            'enable_sql_db': self.enable_sql_db,
            'enable_redis_cache': self.enable_redis_cache,
            'enable_embeddings': self.enable_embeddings,
            'enable_ner': self.enable_ner,
            'enable_metadata_extraction': self.enable_metadata_extraction,
            'enable_cross_reference_detection': self.enable_cross_reference_detection,
            'enable_monitoring': self.enable_monitoring,
            'log_level': self.log_level
        }
    
    def save(self, config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        config_dict = self.to_dict()
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


# Default configurations for different use cases
CONFIGS = {
    'full': PipelineConfig(),
    
    'fast': PipelineConfig(
        batch_size=10,
        enable_embeddings=False,
        enable_ner=False,
        enable_redis_cache=False,
        chunk_size=1000
    ),
    
    'minimal': PipelineConfig(
        batch_size=1,
        enable_cosmos_db=False,
        enable_azure_search=False,
        enable_embeddings=False,
        enable_ner=False,
        enable_metadata_extraction=False,
        enable_cross_reference_detection=False,
        enable_redis_cache=False,
        enable_monitoring=False
    ),
    
    'storage_only': PipelineConfig(
        enable_embeddings=False,
        enable_ner=False,
        enable_metadata_extraction=True,
        enable_cross_reference_detection=True
    )
}


def create_default_config(output_path: str = "pipeline_config.yaml"):
    """Create a default configuration file"""
    config = PipelineConfig()
    config.save(output_path)
    print(f"Created default configuration at: {output_path}")


if __name__ == "__main__":
    # Create default configuration file
    create_default_config()