# Fixed api.py with proper error handling and state management

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import uuid
import json
import os
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# LangSmith imports with error handling
try:
    from langsmith import Client, traceable, get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    traceable = lambda *args, **kwargs: lambda f: f  # Mock decorator
    get_current_run_tree = lambda: None

# Import your graph (using the fixed version)
from graphs import graph, initialize_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# LangSmith initialization with robust error handling
# ==========================
def init_langsmith():
    """Initialize LangSmith with comprehensive error handling"""
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not available - install langsmith package for tracing")
        return None
    
    try:
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        
        if not api_key or not project:
            logger.warning(f"LangSmith config incomplete - API Key: {bool(api_key)}, Project: {bool(project)}")
            return None
        
        # Set environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_API_KEY"] = api_key
        
        client = Client(api_key=api_key, api_url=endpoint)
        
        # Test connection
        list(client.list_projects(limit=1))
        logger.info(f"LangSmith initialized - Project: {project}")
        return client
        
    except Exception as e:
        logger.warning(f"LangSmith initialization failed: {e}")
        return None

def get_trace_url() -> Optional[str]:
    """Safely get trace URL"""
    if not LANGSMITH_AVAILABLE:
        return None
    
    try:
        current_run = get_current_run_tree()
        if current_run and hasattr(current_run, 'id'):
            project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT", "default")
            return f"https://smith.langchain.com/projects/{project}/runs/{current_run.id}"
    except Exception:
        pass
    return None

# ==========================
# Pydantic models
# ==========================
class ResearchBriefRequest(BaseModel):
    topic: str = Field(..., description="Research topic", min_length=3, max_length=200)
    depth: int = Field(default=2, ge=1, le=5, description="Research depth (1-5)")
    audience: str = Field(default="general", description="Target audience")
    follow_up: bool = Field(default=False, description="Follow-up research?")
    user_id: Optional[str] = Field(None, description="User ID")
    parent_session_id: Optional[str] = Field(None, description="Parent session ID")

class ResearchBriefResponse(BaseModel):
    # Session info
    session_id: str
    topic: str
    audience: str
    depth: int
    is_follow_up: bool
    user_id: str
    parent_session_id: Optional[str] = None
    
    # Brief content
    thesis: Optional[str] = None
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    context_summary: Optional[Dict[str, Any]] = None
    created_at: str
    execution_status: str
    
    # Observability
    execution_time_seconds: Optional[float] = None
    total_tokens_used: Optional[int] = None
    errors: List[str] = Field(default_factory=list)
    trace_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"
    langsmith_enabled: bool = False
    graph_compiled: bool = False

# ==========================
# Global variables and metrics
# ==========================
compiled_graph = None
langsmith_client = None
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_execution_time": 0.0,
    "avg_execution_time": 0.0
}

# ==========================
# Graph execution with proper error handling
# ==========================
@traceable(name="research_graph_execution") if LANGSMITH_AVAILABLE else lambda f: f
def execute_research_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute research graph with comprehensive error handling"""
    global compiled_graph, metrics
    
    start_time = time.time()
    
    # Add trace metadata if available
    if LANGSMITH_AVAILABLE:
        try:
            current_run = get_current_run_tree()
            if current_run:
                current_run.update(
                    metadata={
                        "topic": state.get("topic"),
                        "depth": state.get("depth"),
                        "is_follow_up": state.get("follow_up", False),
                        "session_id": state.get("session_id"),
                    }
                )
        except Exception:
            pass  # Ignore tracing errors
    
    try:
        # Ensure state is properly initialized
        state = initialize_state(state)
        
        logger.info(f"Executing research graph for: {state.get('topic')}")
        
        # Execute graph
        if not compiled_graph:
            raise RuntimeError("Graph not compiled")
        
        final_state = compiled_graph.invoke(state)
        
        execution_time = time.time() - start_time
        
        # Update metrics
        metrics["total_execution_time"] += execution_time
        metrics["successful_requests"] += 1
        metrics["avg_execution_time"] = metrics["total_execution_time"] / (
            metrics["successful_requests"] + metrics["failed_requests"]
        )
        
        # Add execution metadata to state
        final_state["api_execution_time"] = execution_time
        
        logger.info(f"Graph execution completed in {execution_time:.2f}s")
        return final_state
        
    except Exception as e:
        execution_time = time.time() - start_time
        metrics["failed_requests"] += 1
        
        logger.error(f"Graph execution failed after {execution_time:.2f}s: {e}")
        
        # Return error state instead of raising
        error_state = initialize_state(state.copy())
        error_state.update({
            "execution_status": "failed",
            "errors": [f"Graph execution failed: {str(e)}"],
            "api_execution_time": execution_time,
            "final_brief": None
        })
        
        return error_state

# ==========================
# FastAPI app setup
# ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph, langsmith_client
    
    try:
        print("Initializing services...")
        
        # Initialize LangSmith
        langsmith_client = init_langsmith()
        
        # Compile graph
        print("Compiling research graph...")
        compiled_graph = graph.compile()
        print("Graph compiled successfully!")
        
        # Test execution
        print("Testing graph execution...")
        test_state = {
            "topic": "Test Topic",
            "depth": 1,
            "audience": "test",
            "user_id": "test",
            "session_id": str(uuid.uuid4()),
            "follow_up": False
        }
        
        test_result = execute_research_graph(test_state)
        if test_result.get("execution_status") == "completed":
            print("Graph test successful!")
        else:
            print(f"Graph test completed with status: {test_result.get('execution_status')}")
            
    except Exception as e:
        print(f"Startup failed: {e}")
        raise
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="Research Brief API",
    description="Generate research briefs with LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# API endpoints
# ==========================
@app.get("/")
def root():
    return {
        "message": "Research Brief API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with service status"""
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
        langsmith_enabled=langsmith_client is not None,
        graph_compiled=compiled_graph is not None
    )

@app.get("/metrics")
def get_metrics():
    """Get application metrics"""
    return {
        "total_requests": metrics["total_requests"],
        "successful_requests": metrics["successful_requests"],
        "failed_requests": metrics["failed_requests"],
        "success_rate": metrics["successful_requests"] / max(1, metrics["total_requests"]),
        "avg_execution_time": round(metrics["avg_execution_time"], 2),
        "langsmith_enabled": langsmith_client is not None
    }

@traceable(name="api_generate_brief") if LANGSMITH_AVAILABLE else lambda f: f
@app.post("/brief", response_model=ResearchBriefResponse)
def generate_research_brief(request: ResearchBriefRequest):
    """
    Generate a research brief
    
    - **topic**: Research topic (3-200 characters)
    - **depth**: Research depth level 1-5 (default: 2)
    - **audience**: Target audience (default: "general")
    - **follow_up**: Whether this builds on previous research
    - **user_id**: User identifier (auto-generated if not provided)
    - **parent_session_id**: Previous session for follow-ups
    """
    global compiled_graph, metrics
    
    # Update request metrics
    metrics["total_requests"] += 1
    
    if not compiled_graph:
        metrics["failed_requests"] += 1
        raise HTTPException(status_code=503, detail="Research graph not available")
    
    api_start_time = time.time()
    
    try:
        # Generate session info
        session_id = str(uuid.uuid4())
        user_id = request.user_id or str(uuid.uuid4())
        
        logger.info(f"Starting research brief - Topic: {request.topic}, User: {user_id[:8]}")
        
        # Create initial state
        state = {
            "topic": request.topic,
            "depth": request.depth,
            "audience": request.audience,
            "follow_up": request.follow_up,
            "user_id": user_id,
            "session_id": session_id,
            "parent_session_id": request.parent_session_id,
            "created_at": datetime.now().isoformat()
        }
        
        # Execute graph
        final_state = execute_research_graph(state)
        
        # Extract results
        brief = final_state.get("final_brief", {})
        context_summary = final_state.get("context_summary")
        execution_status = final_state.get("execution_status", "unknown")
        
        # Calculate timing
        total_execution_time = time.time() - api_start_time
        graph_execution_time = final_state.get("api_execution_time", total_execution_time)
        
        # Get trace URL
        trace_url = get_trace_url()
        
        # Build response
        response = ResearchBriefResponse(
            session_id=session_id,
            topic=request.topic,
            audience=request.audience,
            depth=request.depth,
            is_follow_up=request.follow_up,
            user_id=user_id,
            parent_session_id=request.parent_session_id,
            created_at=state["created_at"],
            execution_status=execution_status,
            
            # Brief content
            thesis=brief.get("thesis"),
            sections=brief.get("sections", []),
            references=brief.get("references", []),
            context_summary=context_summary,
            
            # Observability
            execution_time_seconds=round(graph_execution_time, 2),
            total_tokens_used=final_state.get("total_tokens_used"),
            errors=final_state.get("errors", []),
            trace_url=trace_url
        )
        
        logger.info(f"Research brief completed - Session: {session_id}, "
                   f"Status: {execution_status}, Time: {total_execution_time:.2f}s")
        
        return response
        
    except Exception as e:
        # Update failed metrics
        total_execution_time = time.time() - api_start_time
        metrics["failed_requests"] += 1
        
        logger.error(f"API request failed after {total_execution_time:.2f}s: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate research brief: {str(e)}"
        )

@app.post("/brief/validate")
def validate_request(request: ResearchBriefRequest):
    """Validate a research brief request without executing it"""
    
    errors = []
    warnings = []
    
    # Topic validation
    if len(request.topic.strip()) < 3:
        errors.append("Topic too short (minimum 3 characters)")
    if len(request.topic.strip()) > 200:
        errors.append("Topic too long (maximum 200 characters)")
    
    # Depth validation
    if request.depth < 1 or request.depth > 5:
        errors.append("Depth must be between 1 and 5")
    
    # Follow-up validation
    if request.follow_up and not request.parent_session_id and not request.user_id:
        warnings.append("Follow-up requested but no parent session or user ID provided")
    
    # Performance warnings
    if request.depth >= 4:
        warnings.append("High depth levels may take longer to execute")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "estimated_execution_time": f"{request.depth * 30}-{request.depth * 60} seconds"
    }

@app.get("/test/simple")
def test_simple():
    """Simple test endpoint for basic functionality"""
    try:
        test_state = {
            "topic": "Simple Test",
            "depth": 1,
            "audience": "test",
            "user_id": "test_user",
            "session_id": str(uuid.uuid4()),
            "follow_up": False
        }
        
        result = execute_research_graph(test_state)
        
        return {
            "test_passed": result.get("execution_status") == "completed",
            "execution_status": result.get("execution_status"),
            "errors": result.get("errors", []),
            "execution_time": result.get("api_execution_time", 0),
            "brief_generated": bool(result.get("final_brief"))
        }
        
    except Exception as e:
        return {
            "test_passed": False,
            "error": str(e),
            "message": "Simple test failed"
        }

@app.get("/test/langsmith")
def test_langsmith():
    """Test LangSmith tracing functionality"""
    if not LANGSMITH_AVAILABLE:
        return {
            "langsmith_available": False,
            "message": "LangSmith package not installed"
        }
    
    if not langsmith_client:
        return {
            "langsmith_available": True,
            "langsmith_initialized": False,
            "message": "LangSmith not initialized - check environment variables"
        }
    
    try:
        @traceable(name="langsmith_test")
        def test_function():
            return {
                "test_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "message": "LangSmith test successful"
            }
        
        result = test_function()
        trace_url = get_trace_url()
        
        return {
            "langsmith_available": True,
            "langsmith_initialized": True,
            "test_passed": True,
            "result": result,
            "trace_url": trace_url
        }
        
    except Exception as e:
        return {
            "langsmith_available": True,
            "langsmith_initialized": True,
            "test_passed": False,
            "error": str(e)
        }

# ==========================
# Error handlers
# ==========================
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": exc.errors() if hasattr(exc, 'errors') else str(exc)
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )