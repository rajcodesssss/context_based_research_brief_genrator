from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import json
from datetime import datetime
from contextlib import asynccontextmanager

# Import your existing modules
from graphs import graph
from models import get_history_store, ResearchRequest
from main import convert_to_json_serializable, ensure_session_metadata

# Pydantic models for API requests/responses
class ResearchBriefRequest(BaseModel):
    topic: str = Field(..., description="Research topic", example="AI in Healthcare")
    depth: int = Field(default=2, ge=1, le=5, description="Research depth level (1-5)")
    audience: str = Field(default="general", description="Target audience", example="general")
    follow_up: bool = Field(default=False, description="Is this a follow-up to previous research?")
    user_id: Optional[str] = Field(None, description="User ID for context (auto-generated if not provided)")
    parent_session_id: Optional[str] = Field(None, description="Parent session ID for follow-up research")

class ResearchBriefResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    topic: str = Field(..., description="Research topic")
    audience: str = Field(..., description="Target audience")
    depth: int = Field(..., description="Research depth level")
    is_follow_up: bool = Field(..., description="Whether this was follow-up research")
    thesis: Optional[str] = Field(None, description="Main thesis statement")
    sections: List[Dict[str, Any]] = Field(default=[], description="Research sections")
    references: List[Dict[str, Any]] = Field(default=[], description="Source references")
    context_summary: Optional[Dict[str, Any]] = Field(None, description="Context from previous research (if follow-up)")
    created_at: str = Field(..., description="Creation timestamp")
    user_id: str = Field(..., description="User ID")
    parent_session_id: Optional[str] = Field(None, description="Parent session ID")

class UserHistoryResponse(BaseModel):
    user_id: str
    sessions: List[Dict[str, Any]]
    total_count: int

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"

# Global variables for graph compilation
compiled_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: compile the graph once
    global compiled_graph
    try:
        print("🚀 Compiling research graph...")
        compiled_graph = graph.compile()
        print("✅ Graph compiled successfully!")
    except Exception as e:
        print(f"❌ Failed to compile graph: {e}")
        raise
    
    yield
    
    # Shutdown: cleanup if needed
    print("🔄 Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Context-Aware Research Brief API",
    description="Generate comprehensive research briefs with context-aware follow-up capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
@app.get("/")
def root():
    return {"message": "Welcome to the Context-Aware Research Brief API! Go to /docs for the API docs."}


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.post("/brief", response_model=ResearchBriefResponse)
async def generate_research_brief(request: ResearchBriefRequest):
    """
    Generate a comprehensive research brief
    
    - **topic**: The research topic to investigate
    - **depth**: Research depth level (1=basic, 5=comprehensive)
    - **audience**: Target audience for the brief
    - **follow_up**: Whether this builds on previous research
    - **user_id**: User identifier (auto-generated if not provided)
    - **parent_session_id**: Previous session to build upon (for follow-ups)
    """
    global compiled_graph
    
    if not compiled_graph:
        raise HTTPException(status_code=503, detail="Research graph not initialized")
    
    try:
        # Generate IDs
        session_id = str(uuid.uuid4())
        user_id = request.user_id or str(uuid.uuid4())
        
        # Validate follow-up logic
        if request.follow_up and not request.parent_session_id:
            # Try to get the most recent session for this user
            try:
                history_store = get_history_store()
                user_history = history_store.get_user_history(user_id, limit=1)
                if user_history:
                    request.parent_session_id = user_history[0].id if hasattr(user_history[0], 'id') else None
            except Exception as e:
                print(f"Warning: Could not retrieve user history for follow-up: {e}")
        
        # Create comprehensive initial state
        state = {
            "topic": request.topic,
            "depth": request.depth,
            "audience": request.audience,
            "follow_up": request.follow_up,
            "user_id": user_id,
            "session_id": session_id,
            "parent_session_id": request.parent_session_id,
            "created_at": datetime.now().isoformat(),
            "session_type": "follow_up" if request.follow_up else "initial",
            "is_follow_up": request.follow_up
        }
        
        # Execute the research graph
        final_state = compiled_graph.invoke(state)
        
        # Ensure session metadata is preserved
        final_state = ensure_session_metadata(state, final_state)
        
        # Extract the final brief
        brief_data = None
        if isinstance(final_state, dict) and 'final_brief' in final_state:
            brief_data = final_state['final_brief']
        elif hasattr(final_state, 'final_brief'):
            brief_data = final_state.final_brief
        else:
            brief_data = final_state
        
        if not brief_data:
            raise HTTPException(status_code=500, detail="No research brief generated")
        
        # Convert to JSON serializable format
        if hasattr(brief_data, 'model_dump'):
            try:
                brief_json = brief_data.model_dump(mode="json")
            except Exception:
                brief_json = convert_to_json_serializable(brief_data)
        else:
            brief_json = convert_to_json_serializable(brief_data)
        
        # Extract context summary if available
        context_summary = None
        if request.follow_up:
            context_summary = final_state.get('context_summary') if isinstance(final_state, dict) else getattr(final_state, 'context_summary', None)
            if context_summary:
                context_summary = convert_to_json_serializable(context_summary)
        
        # Build response
        response = ResearchBriefResponse(
            session_id=session_id,
            topic=request.topic,
            audience=request.audience,
            depth=request.depth,
            is_follow_up=request.follow_up,
            created_at=state["created_at"],
            user_id=user_id,
            parent_session_id=request.parent_session_id,
            context_summary=context_summary,
            # Extract brief fields safely
            thesis=brief_json.get('thesis') if isinstance(brief_json, dict) else getattr(brief_json, 'thesis', None),
            sections=brief_json.get('sections', []) if isinstance(brief_json, dict) else getattr(brief_json, 'sections', []),
            references=brief_json.get('references', []) if isinstance(brief_json, dict) else getattr(brief_json, 'references', [])
        )
        
        return response
        
    except Exception as e:
        print(f"Error generating research brief: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate research brief: {str(e)}")

@app.get("/users/{user_id}/history", response_model=UserHistoryResponse)
async def get_user_history(user_id: str, limit: int = 10):
    """
    Get research history for a specific user
    
    - **user_id**: User identifier
    - **limit**: Maximum number of sessions to return (default: 10)
    """
    try:
        history_store = get_history_store()
        user_history = history_store.get_user_history(user_id, limit=limit)
        
        # Convert history to JSON serializable format
        sessions = []
        for session in user_history:
            session_data = convert_to_json_serializable(session)
            sessions.append(session_data)
        
        return UserHistoryResponse(
            user_id=user_id,
            sessions=sessions,
            total_count=len(sessions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user history: {str(e)}")

@app.delete("/users/{user_id}/history")
async def clear_user_history(user_id: str):
    """
    Clear all research history for a specific user
    
    - **user_id**: User identifier
    """
    try:
        history_store = get_history_store()
        # Implement clear_user_history method in your history store
        if hasattr(history_store, 'clear_user_history'):
            history_store.clear_user_history(user_id)
            return {"message": f"History cleared for user {user_id}", "user_id": user_id}
        else:
            raise HTTPException(status_code=501, detail="Clear history functionality not implemented")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear user history: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get details of a specific research session
    
    - **session_id**: Session identifier
    """
    try:
        history_store = get_history_store()
        # Implement get_session method in your history store
        if hasattr(history_store, 'get_session'):
            session = history_store.get_session(session_id)
            if session:
                return convert_to_json_serializable(session)
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            raise HTTPException(status_code=501, detail="Session retrieval not implemented")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")

# Background task for async processing (optional)
@app.post("/brief/async")
async def generate_research_brief_async(request: ResearchBriefRequest, background_tasks: BackgroundTasks):
    """
    Generate a research brief asynchronously (returns immediately with session_id)
    
    Note: This is a placeholder for async processing. You would need to implement
    a job queue system (like Celery, RQ, or similar) for production use.
    """
    session_id = str(uuid.uuid4())
    
    # Add background task (placeholder)
    # background_tasks.add_task(process_research_brief_async, request, session_id)
    
    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Research brief generation started. Use GET /sessions/{session_id} to check status.",
        "estimated_time": "2-5 minutes"
    }

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "status_code": 404
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",  # Assuming this file is named api.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )