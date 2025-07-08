from azure.storage.blob import BlobServiceClient
from typing import List, Dict, Optional, Tuple
import structlog
import hashlib
from datetime import datetime

logger = structlog.get_logger()

class BlobStorageClient:
    """Custom blob storage client that handles connection strings with SAS tokens"""
    
    def __init__(self, connection_string: str, container_name: str):
        # Use BlobServiceClient with the connection string
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.container_name = container_name
        logger.info("Initialized blob storage client", container=container_name)
    
    def list_blobs(self, name_starts_with: str = None) -> List[str]:
        """List blobs in the container"""
        try:
            blobs = []
            for blob in self.container_client.list_blobs(name_starts_with=name_starts_with):
                blobs.append(blob.name)
            return blobs
        except Exception as e:
            logger.error("Failed to list blobs", error=str(e))
            raise
    
    def download_blob(self, blob_name: str) -> bytes:
        """Download a blob"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            logger.error("Failed to download blob", blob_name=blob_name, error=str(e))
            raise
    
    def get_blob_url(self, blob_name: str) -> str:
        """Get the URL for a blob"""
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.url
    
    def get_blob_metadata(self, blob_name: str) -> Dict[str, any]:
        """Get blob metadata including size and last modified"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            
            return {
                "size": properties.size,
                "last_modified": properties.last_modified,
                "content_md5": properties.content_settings.content_md5,
                "etag": properties.etag
            }
        except Exception as e:
            logger.error("Failed to get blob metadata", blob_name=blob_name, error=str(e))
            raise
    
    def download_blob_with_hash(self, blob_name: str) -> Tuple[bytes, str, Dict]:
        """Download blob and calculate SHA256 hash"""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            download_stream = blob_client.download_blob()
            
            # Read content and calculate hash
            content = download_stream.readall()
            sha256_hash = hashlib.sha256(content).hexdigest()
            
            # Get properties
            properties = blob_client.get_blob_properties()
            metadata = {
                "size": properties.size,
                "last_modified": properties.last_modified,
                "etag": properties.etag
            }
            
            return content, sha256_hash, metadata
        except Exception as e:
            logger.error("Failed to download blob with hash", blob_name=blob_name, error=str(e))
            raise
    
    def list_blobs_with_metadata(self, name_starts_with: str = None) -> List[Dict]:
        """List blobs with their metadata"""
        try:
            blobs = []
            for blob in self.container_client.list_blobs(name_starts_with=name_starts_with):
                blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "etag": blob.etag
                })
            return blobs
        except Exception as e:
            logger.error("Failed to list blobs with metadata", error=str(e))
            raise