from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import asyncio
import uuid
from handlers.blob_client import BlobStorageClient
from database.sql_database import sql_db
from handlers.enhanced_document_processor import (
    EnhancedDocumentProcessor, 
    ProcessingStatus,
    ProcessingResult
)
from utils.ai_enrichment_service import ai_enrichment_service
from config.config import settings
import structlog

logger = structlog.get_logger()

class EnhancedPipelineJob:
    """Represents an enhanced document processing pipeline job"""
    def __init__(self, job_id: str, created_by: str):
        self.job_id = job_id
        self.created_by = created_by
        self.created_at = datetime.utcnow()
        self.status = "initializing"
        self.total_documents = 0
        self.processed_documents = 0
        self.failed_documents = 0
        self.documents = []
        self.processing_stats = {
            'total_chunks': 0,
            'total_entities': 0,
            'total_relationships': 0,
            'total_validation_issues': 0,
            'average_processing_time': 0.0
        }

class EnhancedPipelineOrchestrator:
    """Enhanced pipeline orchestrator with comprehensive processing features"""
    
    def __init__(self):
        self.blob_client = BlobStorageClient(
            settings.azure_storage_connection_string,
            settings.azure_storage_container_name
        )
        self.document_processor = EnhancedDocumentProcessor()
        self.db = sql_db
        self.active_jobs = {}
        
        logger.info("Initialized enhanced pipeline orchestrator")
    
    async def start_pipeline_job(self, created_by: str, blob_prefix: Optional[str] = None,
                                specific_blobs: Optional[List[str]] = None,
                                processing_options: Optional[Dict] = None) -> str:
        """
        Start a new enhanced pipeline job
        
        Args:
            created_by: User who initiated the job
            blob_prefix: Optional prefix to filter blobs
            specific_blobs: Optional list of specific blobs to process
            processing_options: Optional processing configuration
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = EnhancedPipelineJob(job_id, created_by)
        self.active_jobs[job_id] = job
        
        try:
            # Create job record
            await self._create_job_record(job)
            
            # Get list of blobs to process
            blobs_to_process = await self._get_blobs_to_process(blob_prefix, specific_blobs)
            job.total_documents = len(blobs_to_process)
            
            # Extract just the names for storage
            blob_names = [blob['name'] for blob in blobs_to_process]
            job.documents = blob_names
            
            logger.info(f"Starting enhanced pipeline job", 
                       job_id=job_id, 
                       total_documents=job.total_documents,
                       processing_options=processing_options)
            
            # Update job status
            job.status = "processing"
            await self._update_job_status(job)
            
            # Process documents asynchronously
            asyncio.create_task(self._process_documents(job, blobs_to_process, processing_options))
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start pipeline job", error=str(e), exc_info=True)
            job.status = "failed"
            await self._update_job_status(job, error=str(e))
            raise
    
    async def _process_documents(self, job: EnhancedPipelineJob, 
                               blobs_to_process: List[Dict],
                               processing_options: Optional[Dict] = None):
        """Process documents with enhanced features"""
        processing_times = []
        
        try:
            # Process each document
            for blob_info in blobs_to_process:
                blob_name = blob_info['name']
                
                try:
                    logger.info(f"Processing document", 
                               job_id=job.job_id, 
                               blob_name=blob_name,
                               blob_size=blob_info.get('size', 0))
                    
                    # Process with enhanced processor
                    result = await self.document_processor.process_document(
                        blob_name, 
                        job.job_id
                    )
                    
                    if result.success:
                        job.processed_documents += 1
                        processing_times.append(result.processing_time)
                        
                        # Update statistics
                        job.processing_stats['total_chunks'] += len(result.chunks)
                        job.processing_stats['total_entities'] += len(result.entity_results.entities)
                        job.processing_stats['total_relationships'] += len(result.relationships)
                        job.processing_stats['total_validation_issues'] += len(result.validation_issues)
                        
                        # Store processing results
                        await self._store_processing_results(job.job_id, blob_name, result)
                        
                        logger.info(f"Document processed successfully",
                                   job_id=job.job_id,
                                   blob_name=blob_name,
                                   chunks=len(result.chunks),
                                   entities=len(result.entity_results.entities),
                                   processing_time=result.processing_time)
                    else:
                        job.failed_documents += 1
                        logger.error(f"Document processing failed",
                                    job_id=job.job_id,
                                    blob_name=blob_name,
                                    error=result.error_message)
                    
                except Exception as e:
                    job.failed_documents += 1
                    logger.error(f"Failed to process document",
                                job_id=job.job_id,
                                blob_name=blob_name,
                                error=str(e),
                                exc_info=True)
                
                # Update job progress
                await self._update_job_status(job)
                
                # Small delay to prevent overload
                await asyncio.sleep(1)
            
            # Calculate average processing time
            if processing_times:
                job.processing_stats['average_processing_time'] = sum(processing_times) / len(processing_times)
            
            # Mark job as completed
            job.status = "completed"
            await self._update_job_status(job)
            
            # Log final statistics
            logger.info(f"Pipeline job completed",
                       job_id=job.job_id,
                       processed=job.processed_documents,
                       failed=job.failed_documents,
                       stats=job.processing_stats)
            
        except Exception as e:
            logger.error(f"Pipeline job failed", 
                        job_id=job.job_id, 
                        error=str(e), 
                        exc_info=True)
            job.status = "failed"
            await self._update_job_status(job, error=str(e))
        
        finally:
            # Clean up
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _store_processing_results(self, job_id: str, blob_name: str, 
                                      result: ProcessingResult):
        """Store enhanced processing results"""
        # Store chunks in Azure Search
        from database.azure_search import AzureSearchManager
        search_manager = AzureSearchManager()
        
        # Convert chunks to search documents
        search_documents = []
        for chunk in result.chunks:
            search_doc = {
                "id": chunk.chunk_id,
                "content": chunk.content,
                "summary": chunk.summary if hasattr(chunk, 'summary') else "",  # ADD SUMMARY HERE
                "document_id": result.metadata.document_id,
                "document_name": result.metadata.document_name,
                "generated_document_name": result.metadata.generated_document_name,
                "chunk_index": chunk.chunk_index,
                "section_path": " > ".join(chunk.section_path),
                "section_title": chunk.section_title,
                "clause_type": chunk.primary_clause_type.value,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "entities": {
                    "actors": chunk.actors,
                    "data_types": chunk.data_types,
                    "time_periods": [tp['period'] for tp in chunk.time_periods],
                    "monetary_amounts": [ma['raw'] for ma in chunk.monetary_amounts]
                },
                "penalties": chunk.penalties,
                "obligations": chunk.obligations,
                "rights": chunk.rights,
                "clause_domain": chunk.clause_domain if hasattr(chunk, 'clause_domain') else [],  # ADD CLAUSE DOMAIN
                "clause_subdomain": chunk.clause_subdomain if hasattr(chunk, 'clause_subdomain') else [],  # ADD CLAUSE SUBDOMAIN
                "internal_refs": [ref['ref'] for ref in chunk.internal_refs],
                "external_refs": [ref['ref'] for ref in chunk.external_refs],
                "token_count": chunk.token_count if hasattr(chunk, 'token_count') else 0,  # ADD TOKEN COUNT
                "metadata": {
                    "document_type": result.metadata.document_type,
                    "regulatory_framework": result.metadata.regulatory_framework,
                    "jurisdiction": result.metadata.jurisdiction,
                    "effective_date": result.metadata.effective_date,
                    "language": chunk.language,
                    "importance_score": chunk.attributes.get('importance_score', 0.5)
                },
                "validation_issues": chunk.validation_result.issues if chunk.validation_result else [],
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Add embeddings if available
            if hasattr(chunk, 'embedding'):
                search_doc['content_vector'] = chunk.embedding
            
            search_documents.append(search_doc)
        
        # Index documents using the correct method
        if search_documents:
            # Convert to format expected by index_document_chunks
            chunk_dicts = []
            for doc in search_documents:
                chunk_dict = {
                    "chunk_id": doc["id"],
                    "document_id": doc["document_id"],
                    "content": doc["content"],
                    "summary": doc.get("summary", ""),  # ADD SUMMARY HERE
                    "section_title": doc["section_title"],
                    "section_path": doc["section_path"].split(" > ") if doc["section_path"] else [],
                    "clause_type": doc["clause_type"],
                    "clause_domain": doc.get("clause_domain", []),  # ADD CLAUSE DOMAIN
                    "clause_subdomain": doc.get("clause_subdomain", []),  # ADD CLAUSE SUBDOMAIN
                    "entities": doc["entities"],
                    "internal_refs": doc["internal_refs"],
                    "external_refs": doc["external_refs"],
                    "penalties": doc["penalties"],
                    "token_count": doc.get("token_count", len(doc["content"].split())),  # Use actual token count if available
                    "start_page": doc.get("start_page"),
                    "end_page": doc.get("end_page")
                }
                
                # Add embedding if available
                if 'content_vector' in doc:
                    chunk_dict['embedding'] = doc['content_vector']
                    
                chunk_dicts.append(chunk_dict)
            
            # Prepare comprehensive document metadata
            logger.debug("Preparing metadata for indexing",
                        doc_type=result.metadata.document_type,
                        framework=result.metadata.regulatory_framework,
                        jurisdiction=result.metadata.jurisdiction,
                        authority=result.metadata.issuing_authority,
                        completeness=result.metadata.completeness_score)
            
            # Debug log the regulation fields
            logger.debug("Document metadata regulation fields",
                        document_name=result.metadata.document_name,
                        regulation=result.metadata.regulation,
                        regulation_normalized=result.metadata.regulation_normalized,
                        regulation_official_name=result.metadata.regulation_official_name,
                        regulation_aliases=result.metadata.regulation_aliases,
                        has_normalized=bool(result.metadata.regulation_normalized),
                        has_official_name=bool(result.metadata.regulation_official_name),
                        has_aliases=bool(result.metadata.regulation_aliases))
            
            doc_metadata = {
                "document_id": result.metadata.document_id,
                "document_name": result.metadata.document_name,
                "generated_document_name": result.metadata.generated_document_name,
                "document_type": result.metadata.document_type,
                "regulatory_framework": result.metadata.regulatory_framework,
                "regulation": result.metadata.regulation,
                "regulation_normalized": result.metadata.regulation_normalized,  # Added
                "regulation_official_name": result.metadata.regulation_official_name,  # Added
                "regulation_aliases": result.metadata.regulation_aliases,  # Added
                "jurisdiction": result.metadata.jurisdiction,
                "territorial_scope": result.metadata.territorial_scope,
                "issuing_authority": result.metadata.issuing_authority,
                "version": result.metadata.version,
                "effective_date": result.metadata.effective_date,
                "enactment_date": result.metadata.enacted_date if hasattr(result.metadata, 'enacted_date') else result.metadata.enactment_date,
                "source_url": result.metadata.source_url,
                "enforcement_authority": result.metadata.enforcement_authority,
                "max_fine_amount": result.metadata.max_fine_amount,
                "max_fine_currency": result.metadata.max_fine_currency,
                "criminal_penalties": result.metadata.criminal_penalties,
                "completeness_score": result.metadata.completeness_score,
                "extraction_confidence": result.metadata.extraction_confidence,
                "language": result.metadata.language,
                "is_official_translation": result.metadata.is_official_translation
            }
            
            await search_manager.index_document_chunks(chunk_dicts, doc_metadata)
        
        # Store metadata in SQL
        metadata_record = {
            "document_id": result.metadata.document_id,
            "job_id": job_id,
            "blob_name": blob_name,
            "document_type": result.metadata.document_type,
            "regulatory_framework": result.metadata.regulatory_framework,
            "jurisdiction": result.metadata.jurisdiction,
            "effective_date": result.metadata.effective_date,
            "total_chunks": len(result.chunks),
            "total_entities": len(result.entity_results.entities),
            "total_relationships": len(result.relationships),
            "validation_issues": len(result.validation_issues),
            "processing_time": result.processing_time,
            "metadata_json": {
                "version": result.metadata.version,
                "issuing_authority": result.metadata.issuing_authority,
                "penalties": result.metadata.penalty_summary,
                "covered_entities": result.metadata.covered_entities,
                "key_obligations": len(result.metadata.key_obligations),
                "quality_scores": {
                    "completeness": result.metadata.completeness_score,
                    "clarity": result.metadata.clarity_score,
                    "structure": result.metadata.structure_score,
                    "confidence": result.metadata.extraction_confidence
                }
            }
        }
        
        self.db.store_document_metadata(metadata_record)
        
        # Store relationships if we have a graph database
        # This would be implemented based on your graph DB choice
        if result.relationships:
            logger.info(f"Storing {len(result.relationships)} relationships",
                       document_id=result.metadata.document_id)
    
    async def _get_blobs_to_process(self, blob_prefix: Optional[str] = None,
                                   specific_blobs: Optional[List[str]] = None) -> List[Dict]:
        """Get list of blobs to process with deduplication"""
        if specific_blobs:
            # Return specific blobs
            return [{'name': blob} for blob in specific_blobs]
        
        # Get all blobs with optional prefix filter
        all_blobs = self.blob_client.list_blobs(name_starts_with=blob_prefix)
        
        # Filter for PDFs
        pdf_blobs = []
        for blob_name in all_blobs:
            if blob_name.lower().endswith('.pdf'):
                # Get metadata for the blob
                try:
                    metadata = self.blob_client.get_blob_metadata(blob_name)
                    pdf_blobs.append({
                        'name': blob_name,
                        'size': metadata.get('size', 0),
                        'last_modified': metadata.get('last_modified')
                    })
                except:
                    # If metadata fails, just use the name
                    pdf_blobs.append({
                        'name': blob_name,
                        'size': 0,
                        'last_modified': None
                    })
        
        # Check for already processed documents
        processed_docs = self.db.get_processed_documents()
        processed_names = {doc['blob_name'] for doc in processed_docs}
        
        # Filter out already processed documents
        new_blobs = [
            blob for blob in pdf_blobs 
            if blob['name'] not in processed_names
        ]
        
        # Also check for documents that failed previously
        failed_docs = self.db.get_failed_documents()
        failed_names = {doc['blob_name'] for doc in failed_docs}
        
        # Include failed documents for retry
        retry_blobs = [
            blob for blob in pdf_blobs 
            if blob['name'] in failed_names
        ]
        
        # Combine new and retry blobs
        blobs_to_process = new_blobs + retry_blobs
        
        # Remove duplicates
        seen = set()
        unique_blobs = []
        for blob in blobs_to_process:
            if blob['name'] not in seen:
                seen.add(blob['name'])
                unique_blobs.append(blob)
        
        logger.info(f"Found blobs to process",
                   total=len(unique_blobs),
                   new=len(new_blobs),
                   retry=len(retry_blobs))
        
        return unique_blobs
    
    async def _create_job_record(self, job: EnhancedPipelineJob):
        """Create job record in database"""
        self.db.create_pipeline_job(
            job_id=job.job_id,
            created_by=job.created_by,
            documents=job.documents
        )
    
    async def _update_job_status(self, job: EnhancedPipelineJob, error: Optional[str] = None):
        """Update job status in database"""
        updates = {
            "status": job.status,
            "total_documents": job.total_documents,
            "processed_documents": job.processed_documents,
            "failed_documents": job.failed_documents,
        }
        
        if error:
            updates["error_details"] = error
            
        if job.status == "completed":
            updates["completed_at"] = datetime.utcnow()
            updates["duration_seconds"] = (datetime.utcnow() - job.created_at).total_seconds()
        
        self.db.update_pipeline_job(job.job_id, updates)
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a specific job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "created_by": job.created_by,
                "created_at": job.created_at.isoformat(),
                "total_documents": job.total_documents,
                "processed_documents": job.processed_documents,
                "failed_documents": job.failed_documents,
                "documents": job.documents,
                "processing_stats": job.processing_stats
            }
        
        # Check database for completed/failed jobs
        return self.db.get_pipeline_job(job_id)
    
    def get_recent_jobs(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline jobs"""
        return self.db.get_recent_pipeline_jobs(limit)