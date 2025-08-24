from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import json
import os
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# LangSmith imports
from langsmith import Client, traceable
from langchain.callbacks import LangChainTracer

# Import your existing modules
from graphs import graph
from models import get_history_store, ResearchRequest
from main import convert_to_json_serializable, ensure_session_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith
def init_langsmith():
    """Initialize LangSmith tracing"""
    try:
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # LANGCHAIN_API_KEY and LANGCHAIN_PROJECT should be set in Render environment
        
        # Initialize LangSmith client
        langsmith_client = Client()
        logger.info("✅ LangSmith initialized successfully")
        return langsmith_client
    except Exception as e:
        logger.warning(f"⚠️ LangSmith initialization failed: {e}")
        return None

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
    # Add observability fields
    execution_time_seconds: Optional[float] = Field(None, description="Total execution time")
    trace_url: Optional[str] = Field(None, description="LangSmith trace URL")

class UserHistoryResponse(BaseModel):
    user_id: str
    sessions: List[Dict[str, Any]]
    total_count: int

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"
    langsmith_enabled: bool = False

class MetricsResponse(BaseModel):
    """Simple metrics endpoint for monitoring"""
    total_requests: int
    avg_execution_time: float
    success_rate: float
    langsmith_traces: int

# Global variables for graph compilation and metrics
compiled_graph = None
langsmith_client = None
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "total_execution_time": 0.0,
    "langsmith_traces": 0
}

@traceable(name="research_graph_execution")
def execute_research_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Traced wrapper for graph execution"""
    global compiled_graph, metrics
    
    start_time = time.time()
    
    try:
        # Execute the research graph
        logger.info(f"🔍 Executing research graph for topic: {state.get('topic')}")
        final_state = compiled_graph.invoke(state)
        
        execution_time = time.time() - start_time
        
        # Update metrics
        metrics["total_execution_time"] += execution_time
        metrics["successful_requests"] += 1
        metrics["langsmith_traces"] += 1
        
        logger.info(f"✅ Graph execution completed in {execution_time:.2f}s")
        
        # Add execution metadata
        if isinstance(final_state, dict):
            final_state["execution_time_seconds"] = execution_time
        else:
            setattr(final_state, "execution_time_seconds", execution_time)
        
        return final_state
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Graph execution failed after {execution_time:.2f}s: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: compile the graph and initialize LangSmith
    global compiled_graph, langsmith_client
    
    try:
        # Initialize LangSmith first
        langsmith_client = init_langsmith()
        
        print("🚀 Compiling research graph...")
        compiled_graph = graph.compile()
        print("✅ Graph compiled successfully!")
        
    except Exception as e:
        print(f"❌ Failed to compile graph: {e}")
        raise
    
    yield
    
    # Shutdown: cleanup if needed
    print("🔄 Shutting down API...")
    if langsmith_client:
        print("📊 LangSmith session completed")

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
    """Health check endpoint with LangSmith status"""
    global langsmith_client
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        langsmith_enabled=langsmith_client is not None
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get basic application metrics"""
    global metrics
    
    avg_execution_time = (
        metrics["total_execution_time"] / metrics["total_requests"] 
        if metrics["total_requests"] > 0 else 0.0
    )
    
    success_rate = (
        metrics["successful_requests"] / metrics["total_requests"] 
        if metrics["total_requests"] > 0 else 0.0
    )
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        avg_execution_time=round(avg_execution_time, 2),
        success_rate=round(success_rate, 2),
        langsmith_traces=metrics["langsmith_traces"]
    )

@traceable(name="generate_research_brief_api")
@app.post("/brief", response_model=ResearchBriefResponse)
async def generate_research_brief(request: ResearchBriefRequest):
    """
    Generate a comprehensive research brief with full observability
    
    - **topic**: The research topic to investigate
    - **depth**: Research depth level (1=basic, 5=comprehensive)
    - **audience**: Target audience for the brief
    - **follow_up**: Whether this builds on previous research
    - **user_id**: User identifier (auto-generated if not provided)
    - **parent_session_id**: Previous session to build upon (for follow-ups)
    """
    global compiled_graph, metrics, langsmith_client
    
    # Update request metrics
    metrics["total_requests"] += 1
    
    if not compiled_graph:
        raise HTTPException(status_code=503, detail="Research graph not initialized")
    
    api_start_time = time.time()
    trace_url = None
    
    try:
        # Generate IDs
        session_id = str(uuid.uuid4())
        user_id = request.user_id or str(uuid.uuid4())
        
        logger.info(f"🎯 Starting research brief generation - Topic: {request.topic}, User: {user_id[:8]}...")
        
        # Validate follow-up logic
        if request.follow_up and not request.parent_session_id:
            # Try to get the most recent session for this user
            try:
                history_store = get_history_store()
                user_history = history_store.get_user_history(user_id, limit=1)
                if user_history:
                    request.parent_session_id = user_history[0].id if hasattr(user_history[0], 'id') else None
                    logger.info(f"🔗 Found parent session for follow-up: {request.parent_session_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve user history for follow-up: {e}")
        
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
        
        # Execute the research graph with tracing
        final_state = execute_research_graph(state)
        
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
        
        # Calculate total execution time
        total_execution_time = time.time() - api_start_time
        graph_execution_time = final_state.get('execution_time_seconds') if isinstance(final_state, dict) else getattr(final_state, 'execution_time_seconds', None)
        
        # Generate trace URL if LangSmith is available
        if langsmith_client:
            try:
                # This is a placeholder - actual trace URL generation depends on LangSmith implementation
                trace_url = f"https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'research-brief-production')}"
            except Exception as e:
                logger.warning(f"Could not generate trace URL: {e}")
        
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
            execution_time_seconds=round(graph_execution_time or total_execution_time, 2),
            trace_url=trace_url,
            # Extract brief fields safely
            thesis=brief_json.get('thesis') if isinstance(brief_json, dict) else getattr(brief_json, 'thesis', None),
            sections=brief_json.get('sections', []) if isinstance(brief_json, dict) else getattr(brief_json, 'sections', []),
            references=brief_json.get('references', []) if isinstance(brief_json, dict) else getattr(brief_json, 'references', [])
        )
        
        logger.info(f"✅ Research brief completed - Session: {session_id}, Time: {total_execution_time:.2f}s")
        
        return response
        
    except Exception as e:
        total_execution_time = time.time() - api_start_time
        logger.error(f"❌ Research brief failed after {total_execution_time:.2f}s - Error: {str(e)}")
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