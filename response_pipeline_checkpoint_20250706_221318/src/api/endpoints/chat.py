"""
Chat API endpoints for the Regulatory Query Agent
"""
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json

from src.models.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationResponse,
    ConversationHistoryResponse,
    StreamingChatResponse
)
from src.models.database import MessageRole
from src.services.auth_service import AuthService
from src.services.conversation_manager import ConversationManager
from src.services.query_manager import QueryManager
from src.services.enhanced_retriever_service import EnhancedRetrieverService
from src.services.response_generator import ResponseGenerator
# Cache import removed
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Dependency for auth
async def get_current_user(
    token: str = Depends(AuthService.verify_token)
) -> Dict:
    """Get current authenticated user"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return token  # Token contains user info


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Send a message and get a response
    
    This endpoint processes user queries through the full pipeline:
    1. Manages conversation context
    2. Analyzes query intent
    3. Retrieves relevant regulatory content
    4. Generates AI response with citations
    """
    try:
        user_id = current_user["user_id"]
        session_id = current_user["session_id"]
        
        # Initialize services
        conversation_manager = ConversationManager()
        query_manager = QueryManager()
        
        retriever = EnhancedRetrieverService()
        response_generator = ResponseGenerator()
        
        # Get or create conversation
        if request.conversation_id:
            conversation = await conversation_manager.get_conversation(
                request.conversation_id
            )
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            # Verify conversation belongs to user
            if conversation["user_id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this conversation"
                )
            # Get conversation_id from dict
            conversation_id = request.conversation_id
        else:
            # Create new conversation
            conversation = await conversation_manager.create_conversation(
                user_id=user_id,
                session_id=session_id,
                title=request.message[:100]  # Use first 100 chars as title
            )
            # Get conversation_id from object
            conversation_id = conversation.conversation_id
        
        # Add user message
        user_message = await conversation_manager.add_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=request.message,
            metadata=request.metadata
        )
        
        # Get conversation history
        history = await conversation_manager.get_conversation_history(
            conversation_id
        )
        
        # Analyze query intent
        # Convert conversation history to format expected by query manager
        history_for_analysis = [
            {"role": msg.role, "content": msg.content}
            for msg in history.messages
        ] if history else []
        
        query_analysis = await query_manager.analyze_query(
            query=request.message,
            conversation_history=history_for_analysis
        )
        
        # Retrieve relevant documents
        search_response = await retriever.retrieve(
            query_analysis=query_analysis,
            top_k=request.max_results or query_analysis.search_filters.get("top_k", 50)
        )
        
        # Generate response
        from src.services.response_generator import GenerationRequest
        generation_request = GenerationRequest(
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            message_id=user_message.message_id,
            query=request.message,
            query_analysis=query_analysis,
            search_results=search_response.results,
            conversation_history=history_for_analysis,  # Use the converted format
            stream=False,
            model="gpt-4",
            temperature=0.0,
            max_tokens=None  # Let response generator use default
        )
        
        response_result = await response_generator.generate(generation_request)
        
        # Add assistant message
        assistant_message = await conversation_manager.add_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=response_result.content,
            metadata={
                "citations": response_result.citations,
                "confidence_score": response_result.confidence_score,
                "intent": query_analysis.primary_intent,
                "tokens_used": response_result.tokens_used
            }
        )
        
        # Update conversation title if it's the first exchange
        if len(history.messages) <= 1:  # Only user message in history
            suggested_title = f"{query_analysis.primary_intent}: {request.message[:50]}"
            await conversation_manager.update_conversation_title(
                conversation_id=conversation_id,
                title=suggested_title
            )
        
        # Transaction handled by service
        
        # Convert EnhancedCitation to Citation format
        from src.models.chat import Citation
        converted_citations = []
        for cit in response_result.citations:
            if hasattr(cit, 'to_dict'):
                cit_dict = cit.to_dict()
            elif isinstance(cit, dict):
                cit_dict = cit
            else:
                cit_dict = cit.__dict__ if hasattr(cit, '__dict__') else {}
            
            citation_obj = Citation(
                text=cit_dict.get('text', ''),
                source=cit_dict.get('source', cit_dict.get('source_file', 'Unknown')),
                page=cit_dict.get('page_number'),
                section=cit_dict.get('article_number') or cit_dict.get('section_number'),
                url=cit_dict.get('url'),
                confidence=cit_dict.get('relevance_score', 0.8)
            )
            converted_citations.append(citation_obj)
        
        # Cache removed - always generate fresh responses
        
        return ChatMessageResponse(
            conversation_id=conversation_id,
            message_id=assistant_message.message_id,
            content=response_result.content,
            citations=converted_citations,
            confidence_score=response_result.confidence_score,
            intent=query_analysis.primary_intent,
            metadata={
                "tokens_used": response_result.tokens_used,
                "model": response_result.model_used
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing message", exc_info=True, error=str(e))
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )


# MCP V2 endpoint removed - using direct pipeline instead
# The /message endpoint provides all necessary functionality


@router.post("/message/stream")
async def send_message_stream(
    request: ChatMessageRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Send a message and get a streaming response
    
    Similar to /message but returns Server-Sent Events (SSE) stream
    """
    try:
        user_id = current_user["user_id"]
        session_id = current_user["session_id"]
        
        # Initialize services
        conversation_manager = ConversationManager()
        query_manager = QueryManager()
        
        retriever = EnhancedRetrieverService()
        response_generator = ResponseGenerator()
        
        # Get or create conversation
        if request.conversation_id:
            conversation = await conversation_manager.get_conversation(
                request.conversation_id
            )
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            # Verify conversation belongs to user
            if conversation["user_id"] != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this conversation"
                )
            conversation_id = request.conversation_id
        else:
            conversation = await conversation_manager.create_conversation(
                user_id=user_id,
                session_id=session_id,
                title=request.message[:100]
            )
            conversation_id = conversation.conversation_id
        
        # Add user message
        user_message = await conversation_manager.add_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=request.message,
            metadata=request.metadata
        )
        
        # Get conversation history
        history = await conversation_manager.get_conversation_history(
            conversation_id
        )
        
        # Analyze query intent
        # Convert conversation history to format expected by query manager
        history_for_analysis = [
            {"role": msg.role, "content": msg.content}
            for msg in history.messages
        ] if history else []
        
        query_analysis = await query_manager.analyze_query(
            query=request.message,
            conversation_history=history_for_analysis
        )
        
        # Retrieve relevant documents
        search_response = await retriever.retrieve(
            query_analysis=query_analysis,
            top_k=request.max_results or query_analysis.search_filters.get("top_k", 50)
        )
        
        async def generate_stream():
            """Generate SSE stream"""
            try:
                # Create generation request for streaming
                from src.services.response_generator import GenerationRequest
                generation_request = GenerationRequest(
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    message_id=user_message.message_id,
                    query=request.message,
                    query_analysis=query_analysis,
                    search_results=search_response.results,
                    conversation_history=history,
                    stream=True,
                    model="gpt-4",
                    temperature=0.0,
                    max_tokens=None  # Let response generator use default
                )
                
                # Start streaming response
                full_content = ""
                async for chunk in response_generator.generate_stream(generation_request):
                    # Send chunk as SSE
                    if chunk["type"] == "content":
                        full_content += chunk["content"]
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif chunk["type"] == "complete":
                        # Save assistant message
                        assistant_message = await conversation_manager.add_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=full_content,
                            metadata={
                                "citations": chunk.get("citations", []),
                                "confidence_score": chunk.get("confidence_score", 0),
                                "intent": query_analysis.primary_intent,
                                "tokens_used": chunk.get("tokens_used", 0)
                            }
                        )
                        # Transaction handled by service
                        
                        # Send final event
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    else:
                        yield f"data: {json.dumps(chunk)}\n\n"
                        
            except Exception as e:
                logger.error("Streaming error", exc_info=True, error=str(e))
                error_chunk = {
                    "type": "error",
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process streaming message"
        )


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    skip: int = 0,
    limit: int = 20,
    active_only: bool = True,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get user's conversations
    
    Returns a paginated list of conversations for the current user
    """
    try:
        user_id = current_user["user_id"]
        conversation_manager = ConversationManager()
        
        conversations = await conversation_manager.get_user_conversations(
            user_id=user_id,
            skip=skip,
            limit=limit,
            active_only=active_only
        )
        
        return [
            ConversationResponse(
                conversation_id=conv["conversation_id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                message_count=conv.get("message_count", 0),
                is_active=conv["is_active"]
            )
            for conv in conversations
        ]
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get conversation history
    
    Returns all messages in a conversation
    """
    try:
        user_id = current_user["user_id"]
        conversation_manager = ConversationManager()
        
        # Verify conversation belongs to user
        conversation = await conversation_manager.get_conversation(
            conversation_id
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        # Verify it belongs to user
        if conversation["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        # Get messages
        messages = await conversation_manager.get_conversation_history(
            conversation_id
        )
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            messages=messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """
    Delete a conversation
    
    Soft deletes a conversation (marks as inactive)
    """
    try:
        user_id = current_user["user_id"]
        conversation_manager = ConversationManager()
        
        # Verify conversation belongs to user
        conversation = await conversation_manager.get_conversation(
            conversation_id
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        # Verify it belongs to user
        if conversation["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        # Soft delete
        await conversation_manager.delete_conversation(conversation_id)
        # Transaction handled by service
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )


@router.post("/conversations/{conversation_id}/feedback")
async def submit_feedback(
    conversation_id: int,
    message_id: int,
    rating: int,
    feedback: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """
    Submit feedback for a message
    
    Allows users to rate responses and provide feedback
    """
    try:
        user_id = current_user["user_id"]
        conversation_manager = ConversationManager()
        
        # Verify conversation belongs to user
        conversation = await conversation_manager.get_conversation(
            conversation_id
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        # Verify it belongs to user
        if conversation["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this conversation"
            )
        
        # Store feedback using conversation manager
        await conversation_manager.update_message_metadata(
            message_id=message_id,
            conversation_id=conversation_id,
            metadata_update={"feedback": {"rating": rating, "feedback": feedback}}
        )
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
        # Transaction handled by service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )