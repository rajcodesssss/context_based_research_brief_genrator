# Fixed graphs.py with better error handling and state management

from __future__ import annotations
import logging
import json
import uuid
import time
from typing import List, Optional, TypedDict, Dict, Any, Union
from langgraph.graph import StateGraph, END

# LangSmith imports for observability
from langsmith import traceable
import os

logger = logging.getLogger(__name__)

# ==========================
# Enhanced Graph state schema with better type definitions
# ==========================
class GraphState(TypedDict, total=False):
    """Enhanced state schema for context-aware research brief graph"""
    # Request information (required)
    topic: str
    depth: int
    audience: str
    user_id: str
    session_id: str
    
    # Request options (optional)
    follow_up: bool
    parent_session_id: Optional[str]
    
    # Processing state (optional)
    context_summary: Optional[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    source_summaries: Optional[List[Dict[str, Any]]]
    final_brief: Optional[Dict[str, Any]]
    
    # Execution tracking (optional)
    search_results: Optional[List[Dict[str, Any]]]
    errors: List[str]  # Always initialize as empty list
    
    # Observability fields (optional)
    node_execution_times: Dict[str, float]  # Always initialize as empty dict
    total_tokens_used: int
    execution_status: str  # 'running', 'completed', 'failed'

# ==========================
# Utility functions for state management
# ==========================
def initialize_state(state: Dict[str, Any]) -> GraphState:
    """Ensure state has all required fields properly initialized"""
    # Initialize tracking fields if not present
    if 'errors' not in state:
        state['errors'] = []
    if 'node_execution_times' not in state:
        state['node_execution_times'] = {}
    if 'total_tokens_used' not in state:
        state['total_tokens_used'] = 0
    if 'execution_status' not in state:
        state['execution_status'] = 'running'
    
    return state

def safe_node_execution(node_name: str):
    """Decorator for safe node execution with comprehensive error handling"""
    def decorator(func):
        @traceable(name=f"node_{node_name}")
        def wrapper(state: GraphState) -> GraphState:
            start_time = time.time()
            
            # Ensure state is properly initialized
            state = initialize_state(state)
            
            logger.info(f"Starting node: {node_name}")
            
            try:
                result = func(state)
                execution_time = time.time() - start_time
                result['node_execution_times'][node_name] = round(execution_time, 2)
                
                logger.info(f"Completed node: {node_name} in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                state['node_execution_times'][node_name] = round(execution_time, 2)
                state['execution_status'] = 'failed'
                error_msg = f"Node {node_name} failed: {str(e)}"
                state['errors'].append(error_msg)
                
                logger.error(f"Failed node: {node_name} after {execution_time:.2f}s - Error: {e}")
                
                # Return state with error information instead of raising
                return state
                
        return wrapper
    return decorator

# ==========================
# Improved node implementations
# ==========================

@safe_node_execution("context_summarization")
def context_summarization_node(state: GraphState) -> GraphState:
    """Summarize user's research history for context-aware follow-ups"""
    print(f"[Node] Context Summarization")
    
    if not state.get('follow_up', False):
        print(" - No context needed for new query")
        state['context_summary'] = None
        return state
    
    try:
        # Mock implementation - replace with actual history store
        context_summary = {
            'previous_topics': [f"Previous topic for user {state['user_id']}"],
            'key_findings': ["Sample finding 1", "Sample finding 2"],
            'knowledge_gaps': ["Gap in research area X"],
            'related_areas': ["Related field Y"]
        }
        
        state['context_summary'] = context_summary
        print(f" - Context summary created with {len(context_summary['previous_topics'])} topics")
        
    except Exception as e:
        print(f" - Context summarization failed: {e}")
        state['context_summary'] = None
        state['errors'].append(f"Context summarization failed: {e}")
    
    return state

@safe_node_execution("planning")
def planning_node(state: GraphState) -> GraphState:
    """Generate research plan"""
    print("[Node] Planning")
    
    try:
        # Mock planning implementation
        plan = {
            'steps': [
                {'order': 1, 'objective': f'Research overview of {state["topic"]}'},
                {'order': 2, 'objective': f'Analyze key aspects of {state["topic"]}'},
                {'order': 3, 'objective': f'Synthesize findings about {state["topic"]}'}
            ],
            'builds_on_previous': state.get('follow_up', False),
            'context_notes': 'Building on previous research' if state.get('context_summary') else None
        }
        
        state['plan'] = plan
        print(f" - Generated plan with {len(plan['steps'])} steps")
        
    except Exception as e:
        print(f" - Planning failed: {e}")
        state['errors'].append(f"Planning failed: {e}")
        # Create minimal fallback plan
        state['plan'] = {
            'steps': [{'order': 1, 'objective': f'Basic research on {state["topic"]}'}],
            'builds_on_previous': False
        }
    
    return state

@safe_node_execution("search")
def search_node(state: GraphState) -> GraphState:
    """Search for sources"""
    print("[Node] Search")
    
    try:
        # Mock search implementation
        search_results = [
            {
                'title': f'Research Article on {state["topic"]}',
                'url': 'https://example.com/article1',
                'snippet': f'This article discusses {state["topic"]} in detail...',
                'content': f'Detailed content about {state["topic"]}. This would normally be fetched content.',
                'source': 'academic'
            },
            {
                'title': f'Industry Report: {state["topic"]}',
                'url': 'https://example.com/report1',
                'snippet': f'Industry insights on {state["topic"]}...',
                'content': f'Industry analysis of {state["topic"]} trends and implications.',
                'source': 'industry'
            }
        ]
        
        state['search_results'] = search_results
        print(f" - Found {len(search_results)} search results")
        
    except Exception as e:
        print(f" - Search failed: {e}")
        state['errors'].append(f"Search failed: {e}")
        state['search_results'] = []
    
    return state

@safe_node_execution("content_processing")
def content_processing_node(state: GraphState) -> GraphState:
    """Process search results into summaries"""
    print("[Node] Content Processing")
    
    search_results = state.get('search_results', [])
    if not search_results:
        print(" - No search results to process")
        state['source_summaries'] = []
        return state
    
    try:
        summaries = []
        for i, result in enumerate(search_results):
            summary = {
                'source_id': f'source_{i+1}',
                'url': result.get('url', ''),
                'title': result.get('title', 'Untitled'),
                'key_points': [
                    f'Key point 1 about {state["topic"]}',
                    f'Key point 2 about {state["topic"]}',
                    f'Important finding regarding {state["topic"]}'
                ],
                'relevance_score': 0.85,
                'summary': f'Summary of content about {state["topic"]}'
            }
            summaries.append(summary)
        
        state['source_summaries'] = summaries
        state['total_tokens_used'] += len(summaries) * 100  # Mock token usage
        print(f" - Created {len(summaries)} source summaries")
        
    except Exception as e:
        print(f" - Content processing failed: {e}")
        state['errors'].append(f"Content processing failed: {e}")
        state['source_summaries'] = []
    
    return state

@safe_node_execution("synthesis")
def synthesis_node(state: GraphState) -> GraphState:
    """Synthesize final brief from summaries"""
    print("[Node] Synthesis")
    
    if not state.get('plan') or not state.get('source_summaries'):
        error_msg = "Cannot synthesize: missing plan or source summaries"
        print(f" - {error_msg}")
        state['errors'].append(error_msg)
        return state
    
    try:
        # Create comprehensive final brief
        final_brief = {
            'topic': state['topic'],
            'audience': state['audience'],
            'depth': state['depth'],
            'thesis': f'Based on current research, {state["topic"]} represents a significant area of development with multiple implications for {state["audience"]}.',
            'sections': [
                {
                    'heading': 'Overview',
                    'content': f'This section provides an overview of {state["topic"]}. Current research indicates several key trends and developments in this area.'
                },
                {
                    'heading': 'Key Findings',
                    'content': f'Research on {state["topic"]} reveals important insights:\n• Finding 1 related to {state["topic"]}\n• Finding 2 about market trends\n• Finding 3 regarding future implications'
                },
                {
                    'heading': 'Analysis',
                    'content': f'Analysis of {state["topic"]} shows complex interactions between various factors. The research depth of {state["depth"]} allows for comprehensive examination.'
                },
                {
                    'heading': 'Conclusions',
                    'content': f'In conclusion, {state["topic"]} is a rapidly evolving field with significant implications for {state["audience"]}.'
                }
            ],
            'references': [
                {
                    'source_id': summary['source_id'],
                    'url': summary['url'],
                    'title': summary['title']
                }
                for summary in state.get('source_summaries', [])
            ],
            'generated_at': time.time(),
            'session_id': state['session_id'],
            'is_follow_up': state.get('follow_up', False),
            'parent_session_id': state.get('parent_session_id')
        }
        
        state['final_brief'] = final_brief
        state['total_tokens_used'] += 500  # Mock synthesis token usage
        print(f" - Generated brief with {len(final_brief['sections'])} sections")
        
    except Exception as e:
        print(f" - Synthesis failed: {e}")
        state['errors'].append(f"Synthesis failed: {e}")
        # Create minimal fallback brief
        state['final_brief'] = {
            'topic': state['topic'],
            'audience': state['audience'],
            'depth': state['depth'],
            'thesis': f'Brief overview of {state["topic"]}.',
            'sections': [{'heading': 'Summary', 'content': f'Basic information about {state["topic"]}.'}],
            'references': [],
            'generated_at': time.time(),
            'session_id': state['session_id'],
            'is_follow_up': state.get('follow_up', False)
        }
    
    return state

@safe_node_execution("validation")
def validation_node(state: GraphState) -> GraphState:
    """Validate and finalize the research brief"""
    print("[Node] Validation")
    
    if not state.get('final_brief'):
        error_msg = "No final brief to validate"
        print(f" - {error_msg}")
        state['errors'].append(error_msg)
        state['execution_status'] = 'failed'
        return state
    
    try:
        brief = state['final_brief']
        validation_errors = []
        
        # Validation checks
        if not brief.get('thesis') or len(brief['thesis'].strip()) < 10:
            validation_errors.append("Thesis too short or missing")
        
        if not brief.get('sections') or len(brief['sections']) == 0:
            validation_errors.append("No sections in brief")
        
        if validation_errors:
            print(f" - Validation warnings: {'; '.join(validation_errors)}")
            state['errors'].extend(validation_errors)
        else:
            print(" - Brief validation passed")
        
        # Mark as completed
        state['execution_status'] = 'completed'
        
        # Calculate total execution time
        total_time = sum(state.get('node_execution_times', {}).values())
        print(f" - Total execution time: {total_time:.2f}s")
        print(f" - Total tokens used: {state.get('total_tokens_used', 0)}")
        
        # Log any errors
        if state.get('errors'):
            print(f" - Total errors: {len(state['errors'])}")
            for error in state['errors']:
                print(f"   ! {error}")
        
    except Exception as e:
        print(f" - Validation failed: {e}")
        state['errors'].append(f"Validation failed: {e}")
        state['execution_status'] = 'failed'
    
    return state

# ==========================
# Conditional routing function
# ==========================
def should_continue(state: GraphState) -> str:
    """Determine if execution should continue based on state"""
    
    # Check for critical errors that should stop execution
    critical_errors = [error for error in state.get('errors', []) if 'failed' in error.lower()]
    
    if critical_errors:
        logger.warning(f"Critical errors detected, stopping execution: {critical_errors}")
        return END
    
    # Check execution status
    if state.get('execution_status') == 'failed':
        return END
    
    # Normal continuation
    return "validation"

# ==========================
# Build improved graph
# ==========================
@traceable(name="create_research_graph")
def create_graph() -> StateGraph:
    """Create and configure the improved research brief graph"""
    
    # Create graph with enhanced state schema
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("context_summarization", context_summarization_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("content_processing", content_processing_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("validation", validation_node)
    
    # Set entry point
    workflow.set_entry_point("context_summarization")
    
    # Add sequential edges
    workflow.add_edge("context_summarization", "planning")
    workflow.add_edge("planning", "search")
    workflow.add_edge("search", "content_processing")
    workflow.add_edge("content_processing", "synthesis")
    
    # Add conditional edge for error handling
    workflow.add_conditional_edges(
        "synthesis",
        should_continue,
        {
            "validation": "validation",
            END: END
        }
    )
    
    # Validation always ends
    workflow.add_edge("validation", END)
    
    return workflow

# Create the graph instance
graph = create_graph()

# ==========================
# Test functions
# ==========================
@traceable(name="test_graph_execution")
def test_graph_execution(topic: str = "AI in Healthcare", depth: int = 2):
    """Test the improved graph"""
    print(f"\n=== Testing Improved Graph ===")
    print(f"Topic: {topic}, Depth: {depth}")
    print("=" * 40)
    
    state = {
        "topic": topic,
        "depth": depth,
        "audience": "general",
        "follow_up": False,
        "user_id": "test_user",
        "session_id": str(uuid.uuid4())
    }
    
    try:
        compiled_graph = graph.compile()
        final_state = compiled_graph.invoke(state)
        
        print(f"\n=== Execution Complete ===")
        print(f"Status: {final_state.get('execution_status', 'unknown')}")
        print(f"Errors: {len(final_state.get('errors', []))}")
        
        if final_state.get('final_brief'):
            brief = final_state['final_brief']
            print(f"Brief sections: {len(brief.get('sections', []))}")
            print(f"References: {len(brief.get('references', []))}")
        
        return final_state
        
    except Exception as e:
        print(f"Graph execution failed: {e}")
        return None

if __name__ == "__main__":
    test_graph_execution()