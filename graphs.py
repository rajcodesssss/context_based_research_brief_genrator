# graphs.py - Updated with context-aware execution for Increment 4
"""
Updated LangGraph implementation with context-aware execution and user history support.
"""

from __future__ import annotations
import logging
import json
import uuid
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph

# Import your existing models (updated)
from models import (
    ResearchPlan, SourceSummary, FinalBrief, ContextSummary, 
    ResearchRequest, get_history_store, make_mock_plan,
    make_mock_source_summaries, compile_final_brief
)

# Import the LLM and search functionality
from llm.providers import (
    create_research_plan_with_llm,
    create_source_summary_with_llm, 
    synthesize_final_brief_with_llm
)
from tools.search import search_for_topic

logger = logging.getLogger(__name__)

# ==========================
# Enhanced Graph state schema with context support
# ==========================
class GraphState(TypedDict):
    """Enhanced state schema for context-aware research brief graph"""
    # Request information
    topic: str
    depth: int
    audience: str
    follow_up: bool
    user_id: str
    session_id: Optional[str]
    parent_session_id: Optional[str]
    
    # Processing state
    context_summary: Optional[ContextSummary]
    plan: Optional[ResearchPlan]
    source_summaries: Optional[List[SourceSummary]]
    final_brief: Optional[FinalBrief]
    
    # Execution tracking
    search_results: Optional[List[dict]]
    errors: Optional[List[str]]

# ==========================
# New Context Summarization Node
# ==========================
def context_summarization_node(state: GraphState) -> GraphState:
    """Summarize user's research history for context-aware follow-ups"""
    print("[Node] Context Summarization")
    
    if not state.get('follow_up', False):
        print(" - No context needed for new query")
        state['context_summary'] = None
        return state
    
    try:
        # Get user history from storage
        history_store = get_history_store()
        user_history = history_store.get_user_history(state['user_id'], limit=5)
        
        if not user_history:
            print(" - No previous research history found")
            state['context_summary'] = None
            return state
        
        print(f" - Found {len(user_history)} previous research sessions")
        
        # Extract context information
        previous_topics = []
        key_findings = []
        knowledge_gaps = []
        related_areas = []
        
        for session in user_history:
            # Extract topic
            previous_topics.append(session['topic'])
            
            # Extract findings from brief
            brief = session['final_brief']
            
            # Look for key findings in sections
            for section in brief.get('sections', []):
                if 'finding' in section.get('heading', '').lower():
                    # Extract bullet points from content
                    content = section.get('content', '')
                    findings = [line.strip('• ') for line in content.split('\n') if line.strip().startswith('•')]
                    key_findings.extend(findings[:2])  # Limit to 2 findings per session
            
            # Extract knowledge gaps (mock implementation)
            if 'conclusion' in str(brief).lower():
                knowledge_gaps.append(f"Further research needed on {session['topic']}")
            
            # Add related areas based on topic similarity
            current_topic_words = set(state['topic'].lower().split())
            session_topic_words = set(session['topic'].lower().split())
            if current_topic_words & session_topic_words:  # If there's overlap
                related_areas.append(session['topic'])
        
        # Create context summary
        context_summary = ContextSummary(
            previous_topics=list(set(previous_topics))[:5],  # Unique topics, max 5
            key_findings=list(set(key_findings))[:10],  # Unique findings, max 10
            knowledge_gaps=list(set(knowledge_gaps))[:5],  # Unique gaps, max 5
            related_areas=list(set(related_areas))[:5]  # Unique areas, max 5
        )
        
        state['context_summary'] = context_summary
        
        print(f" - Context summary created:")
        print(f"   Previous topics: {len(context_summary.previous_topics)}")
        print(f"   Key findings: {len(context_summary.key_findings)}")
        print(f"   Knowledge gaps: {len(context_summary.knowledge_gaps)}")
        print(f"   Related areas: {len(context_summary.related_areas)}")
        
    except Exception as e:
        logger.error(f"Context summarization failed: {e}")
        print(f" - Context summarization failed: {e}")
        state['context_summary'] = None
        
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append(f"Context summarization failed: {e}")
    
    return state

# ==========================
# Updated Planning Node with Context Awareness
# ==========================
def planning_node(state: GraphState) -> GraphState:
    """Generate research plan using LLM with context awareness"""
    print("[Node] Context-Aware Planning with LLM")
    
    try:
        context_summary = state.get('context_summary')
        is_follow_up = state.get('follow_up', False)
        
        # For follow-up queries, incorporate context into planning
        if is_follow_up and context_summary:
            print(" - Incorporating context from previous research")
            
            # Create context-aware research plan
            # Note: You'll need to update your create_research_plan_with_llm function
            # to accept context parameters, or handle it here with mock logic
            try:
                # Try to use context-aware LLM planning (requires LLM update)
                plan = create_research_plan_with_llm(
                    topic=state['topic'],
                    depth=state['depth'],
                    # If your LLM function supports context, pass it here
                    # context_summary=context_summary
                )
            except TypeError:
                # Fallback if LLM function doesn't support context yet
                plan = create_research_plan_with_llm(
                    topic=state['topic'],
                    depth=state['depth']
                )
                # Manually add context awareness
                plan.builds_on_previous = True
                plan.context_notes = f"Building on research: {', '.join(context_summary.previous_topics[:3])}"
            
        else:
            # Regular planning for new queries
            plan = create_research_plan_with_llm(
                topic=state['topic'],
                depth=state['depth']
            )
        
        state['plan'] = plan
        print(f" - Generated {'context-aware' if is_follow_up else 'new'} plan with {len(plan.steps)} steps")
        
        # Log the plan steps
        for step in plan.steps:
            print(f"   Step {step.order}: {step.objective}")
        
        if hasattr(plan, 'context_notes') and plan.context_notes:
            print(f"   Context: {plan.context_notes}")
            
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        print(f" - Planning failed: {e}")
        
        # Fallback to mock plan with context awareness
        context_aware = state.get('follow_up', False) and state.get('context_summary') is not None
        state['plan'] = make_mock_plan(state['topic'], state['depth'], context_aware=context_aware)
        
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append(f"Planning with LLM failed: {e}")
    
    return state

# ==========================
# Other nodes remain the same but with enhanced logging
# ==========================
def search_node(state: GraphState) -> GraphState:
    """Search for sources using web search tools"""
    print("[Node] Web Search")
    
    context_summary = state.get('context_summary')
    if context_summary:
        print(f" - Context-aware search (building on {len(context_summary.previous_topics)} previous topics)")
    
    try:
        # Perform web search
        search_results = search_for_topic(
            topic=state['topic'],
            depth=state['depth']
        )
        
        state['search_results'] = search_results
        print(f" - Found {len(search_results)} search results")
        
        # Log search results
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. {result.get('title', 'Untitled')[:60]}...")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f" - Search failed: {e}")
        
        # Create mock search results as fallback
        state['search_results'] = [
            {
                'title': f"Mock Search Result for {state['topic']}",
                'url': 'https://example.com/mock',
                'snippet': f"Mock content about {state['topic']} for testing purposes.",
                'content': f"This is mock content about {state['topic']}. In a real implementation, this would contain actual web search results and fetched content.",
                'source': 'mock'
            }
        ]
        
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append(f"Web search failed: {e}")
    
    return state

def content_fetching_node(state: GraphState) -> GraphState:
    """Process search results into source summaries using LLM"""
    print("[Node] Content Processing & Summarization")
    
    if not state.get('search_results'):
        print(" - No search results to process")
        state['source_summaries'] = []
        return state
    
    summaries = []
    
    try:
        for i, search_result in enumerate(state['search_results'], 1):
            print(f" - Processing source {i}/{len(state['search_results'])}")
            
            # Create source summary using LLM
            summary = create_source_summary_with_llm(
                source_id=f"source_{i}",
                url=search_result.get('url', ''),
                title=search_result.get('title', 'Untitled'),
                content=search_result.get('content', search_result.get('snippet', '')),
                topic=state['topic']
            )
            
            summaries.append(summary)
            print(f"   Created summary with {len(summary.key_points)} key points")
        
        state['source_summaries'] = summaries
        print(f" - Created {len(summaries)} source summaries")
        
    except Exception as e:
        logger.error(f"Content processing failed: {e}")
        print(f" - Content processing failed: {e}")
        
        # Fallback to mock summaries
        state['source_summaries'] = make_mock_source_summaries(
            state['topic'], 
            state['depth']
        )
        
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append(f"Content processing failed: {e}")
    
    return state

def synthesis_node(state: GraphState) -> GraphState:
    """Synthesize final brief from source summaries using LLM with context awareness"""
    print("[Node] Context-Aware Synthesis with LLM")
    
    if not state.get('plan') or not state.get('source_summaries'):
        print(" - Missing plan or source summaries for synthesis")
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append("Cannot synthesize: missing plan or source summaries")
        return state
    
    try:
        context_summary = state.get('context_summary')
        is_follow_up = state.get('follow_up', False)
        
        if is_follow_up and context_summary:
            print(f" - Incorporating context from {len(context_summary.previous_topics)} previous research sessions")
        
        # Try to use context-aware synthesis
        try:
            final_brief = synthesize_final_brief_with_llm(
                topic=state['topic'],
                depth=state['depth'],
                audience=state['audience'],
                plan=state['plan'],
                summaries=state['source_summaries'],
                # Pass context if your function supports it
                # context_summary=context_summary if is_follow_up else None
            )
        except TypeError:
            # Fallback if LLM function doesn't support context yet
            final_brief = synthesize_final_brief_with_llm(
                topic=state['topic'],
                depth=state['depth'],
                audience=state['audience'],
                plan=state['plan'],
                summaries=state['source_summaries']
            )
        
        # Add context metadata to the brief
        if hasattr(final_brief, '__dict__'):
            final_brief.session_id = state.get('session_id')
            final_brief.is_follow_up = is_follow_up
            final_brief.parent_session_id = state.get('parent_session_id')
        
        state['final_brief'] = final_brief
        print(f" - Generated {'context-aware' if is_follow_up else 'new'} brief with {len(final_brief.sections)} sections")
        print(f" - Brief thesis: {final_brief.thesis[:100]}...")
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        print(f" - Synthesis failed: {e}")
        
        # Fallback to mock compilation with context awareness
        final_brief = compile_final_brief(
            topic=state['topic'],
            depth=state['depth'], 
            audience=state['audience'],
            summaries=state['source_summaries'],
            session_id=state.get('session_id'),
            is_follow_up=is_follow_up,
            parent_session_id=state.get('parent_session_id'),
            context_summary=context_summary
        )
        state['final_brief'] = final_brief
        
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append(f"LLM synthesis failed: {e}")
    
    return state

def post_processing_node(state: GraphState) -> GraphState:
    """Post-process, validate, and save the final brief with context tracking"""
    print("[Node] Post-processing, Validation & Storage")
    
    if not state.get('final_brief'):
        print(" - No final brief to post-process")
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append("No final brief generated")
        return state
    
    try:
        brief = state['final_brief']
        
        # Basic validation checks
        validation_errors = []
        
        if not brief.thesis or len(brief.thesis.strip()) < 10:
            validation_errors.append("Thesis too short or missing")
        
        if not brief.sections or len(brief.sections) == 0:
            validation_errors.append("No sections in brief")
        
        if not brief.references or len(brief.references) == 0:
            validation_errors.append("No references in brief")
        
        # Check section content
        for i, section in enumerate(brief.sections):
            if not section.content or len(section.content.strip()) < 20:
                validation_errors.append(f"Section {i+1} content too short")
        
        if validation_errors:
            print(f" - Validation warnings: {'; '.join(validation_errors)}")
            if not state.get('errors'):
                state['errors'] = []
            state['errors'].extend([f"Validation: {err}" for err in validation_errors])
        else:
            print(" - Brief validation passed")
        
        # Save to user history
        try:
            history_store = get_history_store()
            
            # Convert brief to dict for storage
            if hasattr(brief, 'model_dump'):
                brief_dict = brief.model_dump()
            elif hasattr(brief, 'dict'):
                brief_dict = brief.dict()
            else:
                brief_dict = {
                    'topic': brief.topic,
                    'audience': brief.audience,
                    'depth': brief.depth,
                    'thesis': brief.thesis,
                    'sections': [{'heading': s.heading, 'content': s.content} for s in brief.sections],
                    'references': [{'source_id': r.source_id, 'url': str(r.url) if r.url else None, 'title': r.title} for r in brief.references],
                    'generated_at': brief.generated_at.isoformat() if hasattr(brief.generated_at, 'isoformat') else str(brief.generated_at)
                }
            
            # Save the research session
            history_store.save_brief(
                user_id=state['user_id'],
                session_id=state['session_id'],
                topic=state['topic'],
                final_brief=brief_dict,
                is_follow_up=state.get('follow_up', False),
                parent_session_id=state.get('parent_session_id')
            )
            
            print(f" - 💾 Saved research session to user history")
            
        except Exception as e:
            logger.error(f"Failed to save to history: {e}")
            print(f" - ⚠️ Failed to save to history: {e}")
            if not state.get('errors'):
                state['errors'] = []
            state['errors'].append(f"History save failed: {e}")
        
        # Add execution summary
        print(f" - Final brief summary:")
        print(f"   Topic: {brief.topic}")
        print(f"   Sections: {len(brief.sections)}")
        print(f"   References: {len(brief.references)}")
        print(f"   Audience: {brief.audience}")
        print(f"   Depth: {brief.depth}")
        print(f"   Follow-up: {state.get('follow_up', False)}")
        print(f"   Session ID: {state.get('session_id', 'N/A')}")
        
        # Log any errors that occurred during processing
        if state.get('errors'):
            print(f" - ⚠️ Errors during processing: {len(state['errors'])}")
            for error in state['errors']:
                print(f"   ! {error}")
        
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        print(f" - Post-processing failed: {e}")
        
        if not state.get('errors'):
            state['errors'] = []
        state['errors'].append(f"Post-processing failed: {e}")
    
    return state

# ==========================
# Build updated graph with context routing
# ==========================
def create_graph() -> StateGraph:
    """Create and configure the context-aware research brief graph"""
    
    # Create graph with enhanced state schema
    graph = StateGraph(GraphState)
    
    # Add nodes including new context node
    graph.add_node("context_summarization", context_summarization_node)
    graph.add_node("planning", planning_node)
    graph.add_node("search", search_node)
    graph.add_node("content_fetching", content_fetching_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("post_processing", post_processing_node)
    
    # Set entry point to context summarization
    graph.set_entry_point("context_summarization")
    
    # Add edges (define execution order)
    graph.add_edge("context_summarization", "planning")
    graph.add_edge("planning", "search")
    graph.add_edge("search", "content_fetching")
    graph.add_edge("content_fetching", "synthesis")
    graph.add_edge("synthesis", "post_processing")
    
    # Set finish point
    graph.set_finish_point("post_processing")
    
    return graph

# Create the graph instance
graph = create_graph()

# ==========================
# Enhanced utility functions for testing context features
# ==========================
def test_context_aware_execution():
    """Test the context-aware features with a sequence of related queries"""
    print(f"\n=== Testing Context-Aware Execution ===")
    print("=" * 50)
    
    user_id = "test_user_context"
    
    # First query - establish context
    print("\n🔍 QUERY 1: Initial research")
    session_1_id = str(uuid.uuid4())
    
    state_1 = {
        "topic": "Renewable Energy Trends 2024",
        "depth": 3,
        "audience": "business leaders",
        "follow_up": False,
        "user_id": user_id,
        "session_id": session_1_id,
        "parent_session_id": None
    }
    
    try:
        compiled_graph = graph.compile()
        final_state_1 = compiled_graph.invoke(state_1)
        
        if final_state_1.get('final_brief'):
            print("✅ First research session completed")
            print(f"   Generated brief with {len(final_state_1['final_brief'].sections)} sections")
        else:
            print("❌ First research session failed")
            
    except Exception as e:
        print(f"❌ First query failed: {e}")
        return
    
    # Second query - follow-up research
    print("\n🔍 QUERY 2: Follow-up research (should use context)")
    session_2_id = str(uuid.uuid4())
    
    state_2 = {
        "topic": "Solar Panel Efficiency Improvements",
        "depth": 3,
        "audience": "business leaders",
        "follow_up": True,  # This should trigger context loading
        "user_id": user_id,
        "session_id": session_2_id,
        "parent_session_id": session_1_id
    }
    
    try:
        final_state_2 = compiled_graph.invoke(state_2)
        
        if final_state_2.get('final_brief'):
            print("✅ Follow-up research session completed")
            print(f"   Generated context-aware brief with {len(final_state_2['final_brief'].sections)} sections")
            
            # Check if context was used
            if final_state_2.get('context_summary'):
                context = final_state_2['context_summary']
                print(f"   📋 Context used: {len(context.previous_topics)} previous topics")
                print(f"   🔍 Previous topics: {', '.join(context.previous_topics)}")
            else:
                print("   ⚠️ No context summary found")
        else:
            print("❌ Follow-up research session failed")
            
    except Exception as e:
        print(f"❌ Follow-up query failed: {e}")
    
    # Display user history
    print("\n📚 USER RESEARCH HISTORY:")
    try:
        history_store = get_history_store()
        user_history = history_store.get_user_history(user_id, limit=10)
        
        print(f"Found {len(user_history)} research sessions:")
        for i, session in enumerate(user_history, 1):
            print(f"  {i}. {session['topic']} ({'follow-up' if session['is_follow_up'] else 'initial'})")
            
    except Exception as e:
        print(f"❌ Failed to retrieve history: {e}")

def test_graph_execution(topic: str = "AI in Healthcare", depth: int = 2, 
                        follow_up: bool = False, user_id: str = "test_user"):
    """Enhanced test function with context support"""
    print(f"\n=== Testing Graph Execution ===")
    print(f"Topic: {topic}")
    print(f"Depth: {depth}")
    print(f"Follow-up: {follow_up}")
    print(f"User ID: {user_id}")
    print("=" * 40)
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    parent_session_id = None
    
    # If follow-up, try to get parent session
    if follow_up:
        try:
            history_store = get_history_store()
            recent = history_store.get_user_history(user_id, limit=1)
            if recent:
                parent_session_id = recent[0]['session_id']
                print(f"📋 Parent session: {parent_session_id}")
        except Exception as e:
            print(f"⚠️ Could not find parent session: {e}")
    
    # Create initial state
    state = {
        "topic": topic,
        "depth": depth,
        "audience": "general",
        "follow_up": follow_up,
        "user_id": user_id,
        "session_id": session_id,
        "parent_session_id": parent_session_id
    }
    
    try:
        # Compile and execute graph
        compiled_graph = graph.compile()
        final_state = compiled_graph.invoke(state)
        
        print("\n=== Execution Complete ===")
        
        # Display results
        if final_state.get('final_brief'):
            brief = final_state['final_brief']
            print(f"✅ Brief generated successfully")
            print(f"  Sections: {len(brief.sections)}")
            print(f"  References: {len(brief.references)}")
            print(f"  Thesis: {brief.thesis[:100]}...")
            
            if hasattr(brief, 'is_follow_up') and brief.is_follow_up:
                print(f"  🔗 This was a follow-up research session")
        else:
            print("❌ No final brief generated")
        
        # Display context information
        if final_state.get('context_summary'):
            context = final_state['context_summary']
            print(f"\n📋 Context Summary:")
            print(f"  Previous topics: {len(context.previous_topics)}")
            print(f"  Key findings: {len(context.key_findings)}")
            print(f"  Knowledge gaps: {len(context.knowledge_gaps)}")
        
        # Display any errors
        if final_state.get('errors'):
            print(f"\n⚠️ Errors encountered: {len(final_state['errors'])}")
            for error in final_state['errors']:
                print(f"  - {error}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ Graph execution failed: {e}")
        return None

if __name__ == "__main__":
    # Test the context-aware functionality
    print("Testing standard execution...")
    test_graph_execution()
    
    print("\n" + "="*60)
    print("Testing context-aware execution...")
    test_context_aware_execution()