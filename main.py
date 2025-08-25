from __future__ import annotations
import json
import os
import uuid
from datetime import datetime
from rich import print as rprint
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.prompt import Prompt, Confirm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pydantic import BaseModel
from typing import Any, Dict, Optional, List, Union

# Import your modules (assuming they exist)
from graphs import graph, GraphState
from models import get_history_store, ResearchRequest

def debug_session_state(state, final_state):
    """Debug function to inspect session state for troubleshooting"""
    rprint("\n[bold yellow]🔍 Debug Information[/bold yellow]")
    
    # Check initial state
    rprint("[blue]Initial State:[/blue]")
    if isinstance(state, dict):
        for key, value in state.items():
            value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            rprint(f"  • {key}: {type(value).__name__} = {value_preview}")
    
    # Check final state
    rprint("\n[blue]Final State Structure:[/blue]")
    rprint(f"  • Type: {type(final_state).__name__}")
    
    if isinstance(final_state, dict):
        rprint("  • Keys:")
        for key, value in final_state.items():
            rprint(f"    - {key}: {type(value).__name__}")
    elif hasattr(final_state, '__dict__'):
        rprint("  • Attributes:")
        for key, value in final_state.__dict__.items():
            rprint(f"    - {key}: {type(value).__name__}")
    
    # Check session_id specifically
    session_id_locations = []
    
    if isinstance(final_state, dict):
        if 'session_id' in final_state:
            session_id = final_state['session_id']
            if session_id:
                session_id_locations.append(f"dict key: {str(session_id)[:8]}...")
    
    if hasattr(final_state, 'session_id'):
        session_id = final_state.session_id
        if session_id:
            session_id_locations.append(f"attribute: {str(session_id)[:8]}...")
    
    if session_id_locations:
        rprint(f"[green]✓ session_id found in: {', '.join(session_id_locations)}[/green]")
    else:
        rprint("[red]✗ session_id not found anywhere in final_state[/red]")

def convert_to_json_serializable(obj):
    """Convert complex objects to JSON serializable format with better HttpUrl handling"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'model_dump'):
        # Pydantic models - handle serialization modes
        try:
            return obj.model_dump(mode="json")
        except Exception as e:
            rprint(f"[yellow]Warning: model_dump failed: {e}[/yellow]")
            # Fallback to dict conversion
            return convert_to_json_serializable(obj.__dict__)
    elif hasattr(obj, '__dict__'):
        # Regular objects with __dict__
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                try:
                    # Special handling for HttpUrl and similar objects
                    if hasattr(value, '__str__') and 'Url' in type(value).__name__:
                        result[key] = str(value)
                    else:
                        result[key] = convert_to_json_serializable(value)
                except Exception as e:
                    # Convert problematic objects to string representation
                    result[key] = str(value)
        return result
    elif hasattr(obj, '__str__'):
        # For HttpUrl and other string-convertible objects
        return str(obj)
    else:
        # Last resort fallback
        return str(obj)

def safe_json_dump(data, file_path):
    """Safely dump data to JSON with proper error handling"""
    try:
        # Convert to JSON serializable format first
        json_data = convert_to_json_serializable(data)
        
        # Test serialization before writing to file
        test_json = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # If test passes, write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(test_json)
        
        return True, None
    except Exception as e:
        return False, str(e)

def get_or_create_user_id():
    """Get existing user ID or create a new one"""
    user_file = "user_profile.txt"
    
    if os.path.exists(user_file):
        with open(user_file, 'r') as f:
            stored_user_id = f.read().strip()
        
        if stored_user_id:  # Check if file is not empty
            use_existing = Confirm.ask(f"Continue as user '{stored_user_id[:8]}...'?", default=True)
            if use_existing:
                return stored_user_id
    
    # Create new user ID
    new_user_id = str(uuid.uuid4())
    
    # Save for future sessions
    with open(user_file, 'w') as f:
        f.write(new_user_id)
    
    rprint(f"[green]👤 Created new user profile: {new_user_id[:8]}...[/green]")
    return new_user_id

def show_user_history(user_id: str):
    """Display user's research history in a nice table"""
    try:
        history_store = get_history_store()
        user_history = history_store.get_user_history(user_id, limit=10)
        
        if not user_history:
            rprint("[dim]📚 No previous research history found.[/dim]")
            return
        
        rprint(f"\n[bold blue]📚 Your Research History ({len(user_history)} sessions)[/bold blue]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="dim", width=12)
        table.add_column("Topic", style="cyan", min_width=30)
        table.add_column("Type", justify="center", width=10)
        table.add_column("Depth", justify="center", width=5)
        
        for session in user_history:
            date = session.created_at.strftime('%Y-%m-%d') if hasattr(session, 'created_at') and session.created_at else 'N/A'
            topic = session.topic[:40] + '...' if len(session.topic) > 40 else session.topic
            session_type = "Follow-up" if getattr(session, 'is_follow_up', False) else "Initial"
            
            # Safely get depth from final_brief
            depth = 'N/A'
            try:
                if hasattr(session, 'final_brief') and session.final_brief:
                    brief_data = json.loads(session.final_brief) if isinstance(session.final_brief, str) else session.final_brief
                    depth = str(brief_data.get('depth', 'N/A'))
            except:
                depth = 'N/A'
            
            table.add_row(date, topic, session_type, depth)
        
        rprint(table)
        
    except Exception as e:
        rprint(f"[red]Error loading history: {e}[/red]")

def display_enhanced_brief(brief_data):
    """Display the research brief in an enhanced, readable format with better error handling"""
    
    # Safe attribute access helper
    def safe_get_attr(obj, attr, default='Unknown'):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        else:
            return getattr(obj, attr, default)
    
    # Brief header
    topic = safe_get_attr(brief_data, 'topic', 'Unknown Topic')
    audience = safe_get_attr(brief_data, 'audience', 'Unknown Audience')
    depth = safe_get_attr(brief_data, 'depth', 'Unknown')
    
    rprint(f"[bold cyan]Topic:[/bold cyan] {topic}")
    rprint(f"[bold cyan]Audience:[/bold cyan] {audience}")
    rprint(f"[bold cyan]Depth Level:[/bold cyan] {depth}/5")
    
    is_follow_up = safe_get_attr(brief_data, 'is_follow_up', False)
    if is_follow_up:
        rprint("[yellow]📎 This is a follow-up research building on previous work[/yellow]")
    
    rprint()
    
    # Thesis
    thesis = safe_get_attr(brief_data, 'thesis', None)
    if thesis:
        rprint("[bold green]📝 Thesis:[/bold green]")
        rprint(Panel(thesis, border_style="green"))
    
    # Sections
    sections = safe_get_attr(brief_data, 'sections', [])
    if sections:
        rprint("\n[bold blue]📚 Research Sections:[/bold blue]")
        for i, section in enumerate(sections, 1):
            heading = safe_get_attr(section, 'heading', f'Section {i}')
            content = safe_get_attr(section, 'content', 'No content available')
            rprint(f"\n[bold]{i}. {heading}[/bold]")
            rprint(content)
    
    # References
    references = safe_get_attr(brief_data, 'references', [])
    if references:
        rprint(f"\n[bold magenta]🔗 References ({len(references)} sources):[/bold magenta]")
        for i, ref in enumerate(references, 1):
            title = safe_get_attr(ref, 'title', "Untitled Source")
            url = safe_get_attr(ref, 'url', "No URL")
            rprint(f"  {i}. {title}")
            rprint(f"     [dim]{url}[/dim]")

def display_execution_summary(final_state, session_id: str, follow_up: bool):
    """Display execution summary and statistics with better error handling"""
    rprint(f"\n[bold]📊 Execution Summary[/bold]")
    rprint(f"Session ID: [dim]{session_id}[/dim]")
    rprint(f"Type: {'Follow-up Research' if follow_up else 'Initial Research'}")
    
    # Show processing steps
    plan = final_state.get('plan') if isinstance(final_state, dict) else getattr(final_state, 'plan', None)
    if plan:
        steps = None
        if isinstance(plan, dict):
            steps = plan.get('steps')
        elif hasattr(plan, 'steps'):
            steps = plan.steps
        
        if steps:
            rprint(f"Research Steps: {len(steps)}")
    
    source_summaries = final_state.get('source_summaries') if isinstance(final_state, dict) else getattr(final_state, 'source_summaries', None)
    if source_summaries:
        rprint(f"Sources Processed: {len(source_summaries)}")
    
    # Show any errors or warnings
    errors = final_state.get('errors', []) if isinstance(final_state, dict) else getattr(final_state, 'errors', [])
    if errors:
        rprint(f"[yellow]⚠️ Issues encountered: {len(errors)}[/yellow]")
        for error in errors:
            rprint(f"  • [dim]{error}[/dim]")
    
    # Check for history save issues specifically
    if any('session_id' in str(error).lower() or 'history' in str(error).lower() for error in errors):
        rprint(f"[red]🔧 History Save Issue Detected[/red]")
        rprint("  • This might be due to missing session_id in the graph state")
        rprint("  • Check that your graph properly passes through session metadata")
        
    if not errors:
        rprint("[green]✅ Research completed successfully![/green]")
    
    # Display final brief summary if available
    brief = final_state.get('final_brief') if isinstance(final_state, dict) else getattr(final_state, 'final_brief', None)
    if brief:
        try:
            # Safe attribute access with multiple fallbacks
            def get_brief_attr(attr, default='Unknown'):
                if isinstance(brief, dict):
                    return brief.get(attr, default)
                elif hasattr(brief, attr):
                    return getattr(brief, attr, default)
                else:
                    return default
            
            topic = get_brief_attr('topic')
            sections = get_brief_attr('sections', [])
            references = get_brief_attr('references', [])
            audience = get_brief_attr('audience')
            depth = get_brief_attr('depth')
            
            sections_count = len(sections) if sections else 0
            refs_count = len(references) if references else 0
            
            rprint(f"\n[bold blue]📋 Final Brief Summary:[/bold blue]")
            rprint(f"  Topic: {topic}")
            rprint(f"  Sections: {sections_count}")
            rprint(f"  References: {refs_count}")
            rprint(f"  Audience: {audience}")
            rprint(f"  Depth: {depth}")
            rprint(f"  Follow-up: {follow_up}")
            rprint(f"  Session ID: {session_id[:8]}..." if session_id else "N/A")
            
        except Exception as e:
            rprint(f"[yellow]Could not parse brief summary: {e}[/yellow]")

def save_research_results(brief_json, save_path, topic):
    """Save research results with enhanced file handling and JSON serialization"""
    try:
        if os.path.isdir(save_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in " -_").strip()
            save_path = os.path.join(save_path, f"{safe_topic}_{timestamp}.json")
        elif not save_path.lower().endswith(".json"):
            save_path += ".json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Use safe JSON dump with serialization handling
        success, error = safe_json_dump(brief_json, save_path)
        
        if success:
            rprint(f"[green]💾 Research brief saved to:[/green] {save_path}")
        else:
            rprint(f"[red]Error saving file: {error}[/red]")
            rprint("[yellow]Attempting fallback save with string conversion...[/yellow]")
            
            # Fallback: convert everything to strings
            fallback_data = convert_to_json_serializable(brief_json)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(fallback_data, f, ensure_ascii=False, indent=2, default=str)
            rprint(f"[green]💾 Fallback save successful:[/green] {save_path}")
        
    except Exception as e:
        rprint(f"[red]Critical error saving file: {e}[/red]")
        rprint("[yellow]Saving as text file instead...[/yellow]")
        
        # Last resort: save as text
        text_path = save_path.replace('.json', '.txt')
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(f"Research Brief - {topic}\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(brief_json))
        
        rprint(f"[green]💾 Saved as text file:[/green] {text_path}")

def ensure_session_metadata(state: dict, final_state) -> dict:
    """Ensure all required session metadata is present in final_state"""
    required_keys = ['session_id', 'user_id', 'topic', 'depth', 'audience', 'follow_up']
    
    if isinstance(final_state, dict):
        # final_state is a dictionary
        for key in required_keys:
            if key not in final_state and key in state:
                final_state[key] = state[key]
                rprint(f"[green]🔧 Added missing {key} to final_state[/green]")
    else:
        # final_state is an object, convert to dict
        final_state_dict = {}
        
        # Copy existing attributes/properties
        if hasattr(final_state, '__dict__'):
            final_state_dict.update(final_state.__dict__)
        
        # Add missing required keys
        for key in required_keys:
            if key not in final_state_dict and key in state:
                final_state_dict[key] = state[key]
                rprint(f"[green]🔧 Added missing {key} to final_state[/green]")
        
        return final_state_dict
    
    return final_state

# Enhanced chatbot function with all fixes
def chatbot():
    """Enhanced chatbot with context-aware research capabilities and error fixes"""
    rprint(Panel.fit("[bold green]Welcome to Context-Aware Research Brief Chatbot[/bold green]"))
    rprint("[dim]Now supports follow-up research building on previous conversations![/dim]\n")

    # 1️⃣ Get or create user ID
    user_id = get_or_create_user_id()
    
    # 2️⃣ Show user's research history (if any)
    show_user_history(user_id)
    
    # 3️⃣ Gather inputs with context awareness
    topic = input("\n📝 Enter research topic: ").strip() or "AI in Healthcare"
    
    # Ask if this is a follow-up to previous research
    follow_up = False
    history_store = get_history_store()
    user_history = history_store.get_user_history(user_id, limit=5)
    
    if user_history:
        rprint(f"\n[yellow]You have {len(user_history)} previous research sessions.[/yellow]")
        follow_up = Confirm.ask("Is this a follow-up to your previous research?", default=False)
        
        if follow_up:
            rprint("[green]✓ This research will build upon your previous work![/green]")
    
    # Other inputs
    depth_input = input("Enter research depth (1–5, default=2): ").strip()
    depth = int(depth_input) if depth_input.isdigit() and 1 <= int(depth_input) <= 5 else 2
    audience = input("Enter target audience (default='general'): ").strip() or "general"
    
    # File save options
    save_path = input("Optional: Enter path to save JSON (press Enter to skip): ").strip() or None

    # 3️⃣ Generate session IDs and determine parent session
    session_id = str(uuid.uuid4())
    parent_session_id = None
    
    if follow_up and user_history:
        # Use the most recent session as parent
        parent_session_id = user_history[0].id if hasattr(user_history[0], 'id') else None
        if parent_session_id:
            rprint(f"[dim]🔗 Building on session: {parent_session_id[:8]}...[/dim]")

    # 4️⃣ Initialize enhanced graph state with all required metadata
    try:
        # Create comprehensive initial state with all session metadata
        state = {
            "topic": topic,
            "depth": depth,
            "audience": audience,
            "follow_up": follow_up,
            "user_id": user_id,
            "session_id": session_id,
            "parent_session_id": parent_session_id,
            # Additional metadata for robust state management
            "created_at": datetime.now().isoformat(),
            "session_type": "follow_up" if follow_up else "initial",
            "is_follow_up": follow_up  # Alternative key name for compatibility
        }
        
        # Validate critical session data is present
        required_fields = ["session_id", "user_id", "topic"]
        missing_fields = [field for field in required_fields if not state.get(field)]
        
        if missing_fields:
            rprint(f"[red]❌ Critical fields missing: {missing_fields}[/red]")
            return
        
        rprint(f"\n[cyan]🚀 Starting {'context-aware' if follow_up else 'new'} research...[/cyan]")
        rprint(f"[dim]Session ID: {session_id[:8]}... | User: {user_id[:8]}...[/dim]")
        
    except Exception as e:
        rprint(f"[red]Error initializing state:[/red] {str(e)}")
        return

    try:
        # 5️⃣ Execute the context-aware graph
        rprint("[yellow]📊 Compiling graph...[/yellow]")
        compiled_graph = graph.compile()
        
        rprint("[yellow]🔄 Executing research workflow...[/yellow]")
        final_state = compiled_graph.invoke(state)

        # 🔧 CRITICAL FIX: Ensure session metadata is preserved
        final_state = ensure_session_metadata(state, final_state)
        
        # Debug the final state
        debug_session_state(state, final_state)

        # 6️⃣ Display context information (if follow-up)
        context_summary = final_state.get('context_summary') if isinstance(final_state, dict) else getattr(final_state, 'context_summary', None)
        if follow_up and context_summary:
            display_context_summary(context_summary)

        # 7️⃣ Display final brief
        rprint(Panel.fit(f"[bold cyan]📋 Research Brief: '{topic}'[/bold cyan]"))
        
        # Handle different possible return formats
        brief_data = None
        if isinstance(final_state, dict) and 'final_brief' in final_state:
            brief_data = final_state['final_brief']
        elif hasattr(final_state, 'final_brief'):
            brief_data = final_state.final_brief
        else:
            brief_data = final_state
        
        # Display the brief with enhanced formatting
        if brief_data:
            if hasattr(brief_data, 'model_dump'):
                display_enhanced_brief(brief_data)
                # Handle JSON serialization with proper conversion
                try:
                    brief_json = brief_data.model_dump(mode="json")
                except Exception as e:
                    rprint(f"[yellow]⚠️ JSON serialization issue, attempting manual conversion: {e}[/yellow]")
                    brief_json = convert_to_json_serializable(brief_data)
            else:
                display_enhanced_brief(brief_data)
                brief_json = convert_to_json_serializable(brief_data)
        else:
            rprint("[red]No final brief generated[/red]")
            return

        # 8️⃣ Show execution summary
        display_execution_summary(final_state, session_id, follow_up)

        # 9️⃣ Save to file if requested
        if save_path:
            save_research_results(brief_json, save_path, topic)

        # 🔟 Ask for follow-up research
        ask_for_follow_up_research(user_id)

    except Exception as e:
        rprint(f"[red]❌ Error executing graph:[/red] {str(e)}")
        rprint("[yellow]Please check your graph implementation and ensure it's properly configured.[/yellow]")
        
        # Debug information
        rprint(f"[dim]Graph type: {type(graph)}[/dim]")

def display_context_summary(context_summary):
    """Display the context summary in a nice format"""
    rprint("\n[bold yellow]📋 Context from Previous Research[/bold yellow]")
    
    # Safe attribute access
    def safe_get(obj, attr, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        else:
            return getattr(obj, attr, default)
    
    previous_topics = safe_get(context_summary, 'previous_topics', [])
    if previous_topics:
        rprint("[blue]Previous Topics:[/blue]")
        for topic in previous_topics:
            rprint(f"  • {topic}")
    
    key_findings = safe_get(context_summary, 'key_findings', [])
    if key_findings:
        rprint("\n[green]Key Findings to Build Upon:[/green]")
        for finding in key_findings[:5]:  # Show top 5
            rprint(f"  • {finding}")
    
    knowledge_gaps = safe_get(context_summary, 'knowledge_gaps', [])
    if knowledge_gaps:
        rprint("\n[red]Knowledge Gaps to Address:[/red]")
        for gap in knowledge_gaps:
            rprint(f"  • {gap}")
    
    rprint()

def ask_for_follow_up_research(user_id: str):
    """Ask user if they want to do follow-up research"""
    rprint("\n" + "="*50)
    
    continue_research = Confirm.ask("Would you like to conduct follow-up research on a related topic?", default=False)
    
    if continue_research:
        rprint("[green]🔄 Starting follow-up research...[/green]")
        rprint("="*50)
        # Recursively call chatbot for follow-up research
        chatbot()
    else:
        rprint("[blue]Thank you for using the Context-Aware Research Assistant![/blue]")
        rprint("[dim]Your research history has been saved for future sessions.[/dim]")

def main():
    """Main function to choose between different modes"""
    rprint(Panel.fit("[bold green]Context-Aware Research Assistant[/bold green]"))
    rprint("[dim]Choose your research mode:[/dim]\n")
    
    try:
        chatbot()
    except KeyboardInterrupt:
        rprint("\n[yellow]Session interrupted by user[/yellow]")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()