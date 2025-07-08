"""
Citation API endpoints - Internal document references only
NO EXTERNAL URLs - All citations must point to PDFs in our index
"""
from typing import Dict, Optional, Any, List
from fastapi import APIRouter, HTTPException, Depends, Path, Query, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import re
import os
from pathlib import Path as PathLib

from src.models.auth import User
from src.api.dependencies import get_current_user, get_optional_user
from src.services.citation_document_service import get_citation_document_service
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.query_manager import QueryManager
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CitationResolveRequest(BaseModel):
    """Request to resolve a citation to its document"""
    citation_text: str = Field(..., description="The citation text to resolve")
    include_metadata: bool = Field(True, description="Include chunk metadata if available")


class CitationDocument(BaseModel):
    """Citation document information"""
    citation_id: str
    citation_text: str
    document_name: str
    jurisdiction: str
    regulation: str
    article_section: Optional[str]
    document_path: str
    document_type: str = "pdf"
    found: bool
    metadata: Optional[Dict[str, Any]] = None


class CitationResolveResponse(BaseModel):
    """Response from citation resolution"""
    found: bool
    citation: Optional[CitationDocument]
    error: Optional[str]


class BatchResolveRequest(BaseModel):
    """Request for batch citation resolution"""
    citations: List[str] = Field(..., description="List of citation texts to resolve")


@router.get("/{citation_id}/document", response_model=CitationDocument)
async def get_citation_document(
    citation_id: str = Path(..., description="The citation ID"),
    current_user: Optional[User] = Depends(get_optional_user)
) -> CitationDocument:
    """
    Get document information for a citation ID.
    
    This endpoint returns the PDF document path and metadata for a citation.
    Citations are ONLY resolved to internal PDF documents, never external URLs.
    """
    try:
        citation_service = get_citation_document_service()
        
        # For demo purposes, we'll reverse-engineer the citation from the ID
        # In production, you'd store this mapping in a database
        
        # Try to find matching citation by checking all jurisdictions
        for jurisdiction_key, config in citation_service.document_patterns.items():
            # Generate test citation text
            test_citations = [
                f"{config['regulation']} Article 1",
                f"{config['regulation']} Section 1",
                f"{config['regulation']}"
            ]
            
            for test_citation in test_citations:
                test_id = citation_service._generate_citation_id(test_citation)
                if test_id == citation_id:
                    doc_info = citation_service.get_document_info(test_citation)
                    if doc_info:
                        return CitationDocument(
                            citation_id=citation_id,
                            citation_text=test_citation,
                            document_name=doc_info["document_name"],
                            jurisdiction=doc_info["jurisdiction"],
                            regulation=doc_info["regulation"],
                            article_section=doc_info["article_section"],
                            document_path=doc_info["internal_path"],
                            document_type="pdf",
                            found=True
                        )
        
        # Citation not found
        logger.warning(f"Citation ID not found: {citation_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Citation with ID '{citation_id}' not found in our document index"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving citation document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve citation document: {str(e)}"
        )


@router.post("/resolve", response_model=CitationResolveResponse)
async def resolve_citation(
    request: CitationResolveRequest,
    current_user: User = Depends(get_current_user)
) -> CitationResolveResponse:
    """
    Resolve a citation text to its source PDF document.
    
    This endpoint takes citation text and returns the internal PDF document
    it references. NO EXTERNAL URLs are returned.
    """
    try:
        citation_service = get_citation_document_service()
        
        # Resolve the citation
        resolution = citation_service.resolve_citation(
            citation_text=request.citation_text,
            chunk_metadata=None  # Could be enhanced with search
        )
        
        if resolution["found"]:
            citation_doc = CitationDocument(
                citation_id=resolution["citation_id"],
                citation_text=request.citation_text,
                document_name=resolution["document_name"],
                jurisdiction=resolution["jurisdiction"],
                regulation=resolution["regulation"],
                article_section=resolution.get("article_section"),
                document_path=resolution["internal_path"],
                document_type="pdf",
                found=True,
                metadata={
                    "chunk_id": resolution.get("chunk_id"),
                    "page_number": resolution.get("page_number"),
                    "clause_type": resolution.get("clause_type"),
                    "clause_title": resolution.get("clause_title")
                } if request.include_metadata else None
            )
            
            return CitationResolveResponse(
                found=True,
                citation=citation_doc
            )
        else:
            return CitationResolveResponse(
                found=False,
                error=resolution.get("error", "Citation could not be resolved to a document")
            )
            
    except Exception as e:
        logger.error(f"Error resolving citation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve citation: {str(e)}"
        )


@router.get("/{citation_id}/pdf")
async def get_citation_pdf(
    citation_id: str = Path(..., description="The citation ID"),
    page: Optional[int] = Query(None, description="Specific page number to return"),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Download or stream the PDF document for a citation.
    
    This endpoint serves the actual PDF file referenced by the citation.
    Optionally specify a page number to get a specific page.
    """
    try:
        # First get the document info
        citation_service = get_citation_document_service()
        
        # Get document path (similar logic to get_citation_document)
        document_path = None
        for jurisdiction_key, config in citation_service.document_patterns.items():
            test_citations = [
                f"{config['regulation']} Article 1",
                f"{config['regulation']} Section 1",
                f"{config['regulation']}"
            ]
            
            for test_citation in test_citations:
                test_id = citation_service._generate_citation_id(test_citation)
                if test_id == citation_id:
                    document_path = config["document_name"]
                    break
            if document_path:
                break
        
        if not document_path:
            raise HTTPException(
                status_code=404,
                detail=f"Citation with ID '{citation_id}' not found"
            )
        
        # In production, you'd have a document storage service
        # For now, we'll return a placeholder response
        logger.info(f"Request to serve PDF: {document_path} (page: {page})")
        
        # Return a JSON response indicating where the PDF would be served from
        # In production, this would use FileResponse or StreamingResponse
        return {
            "message": "PDF serving not implemented in demo",
            "document_path": document_path,
            "page_requested": page,
            "note": "In production, this endpoint would serve the actual PDF file"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serve PDF: {str(e)}"
        )


@router.post("/batch-resolve", response_model=List[CitationResolveResponse])
async def batch_resolve_citations(
    request: BatchResolveRequest,
    current_user: User = Depends(get_current_user)
) -> List[CitationResolveResponse]:
    """
    Resolve multiple citations in a single request.
    
    Useful for processing all citations in a response at once.
    """
    try:
        citation_service = get_citation_document_service()
        results = []
        
        for citation_text in request.citations:
            try:
                resolution = citation_service.resolve_citation(citation_text)
                
                if resolution["found"]:
                    citation_doc = CitationDocument(
                        citation_id=resolution["citation_id"],
                        citation_text=citation_text,
                        document_name=resolution["document_name"],
                        jurisdiction=resolution["jurisdiction"],
                        regulation=resolution["regulation"],
                        article_section=resolution.get("article_section"),
                        document_path=resolution["internal_path"],
                        document_type="pdf",
                        found=True
                    )
                    
                    results.append(CitationResolveResponse(
                        found=True,
                        citation=citation_doc
                    ))
                else:
                    results.append(CitationResolveResponse(
                        found=False,
                        error=f"Citation '{citation_text}' not found"
                    ))
                    
            except Exception as e:
                logger.error(f"Error resolving citation '{citation_text}': {str(e)}")
                results.append(CitationResolveResponse(
                    found=False,
                    error=f"Error processing citation: {str(e)}"
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch citation resolution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve citations: {str(e)}"
        )