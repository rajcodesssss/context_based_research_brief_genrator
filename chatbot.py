#!/usr/bin/env python3
"""
Complete standalone chatbot with API integration option
This file can work independently or integrate with your FastAPI server
"""

import requests
import json
import os
import uuid
from datetime import datetime
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from typing import Dict, Any, List, Optional

# API Configuration
API_CONFIG = {
    "enabled": False,  # Set to True to use API instead of direct graph execution
    "base_url": "http://localhost:8000",  # Your API base URL
    "timeout": 300  # 5 minutes timeout for API calls
}

# ===============================
# UTILITY FUNCTIONS
# ===============================

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

def convert_to_json_serializable(obj):
    """Convert complex objects to JSON serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return convert_to_json_serializable(obj.__dict__)
    elif hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                try:
                    if hasattr(value, '__str__') and 'Url' in type(value).__name__:
                        result[key] = str(value)
                    else:
                        result[key] = convert_to_json_serializable(value)
                except Exception:
                    result[key] = str(value)
        return result
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        return str(obj)

def save_research_results(brief_data, save_path: str, topic: str):
    """Save research results to JSON file"""
    try:
        if os.path.isdir(save_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in " -_").strip()
            save_path = os.path.join(save_path, f"{safe_topic}_{timestamp}.json")
        elif not save_path.lower().endswith(".json"):
            save_path += ".json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Convert to JSON serializable format
        json_data = convert_to_json_serializable(brief_data)
        
        # Write to file
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        rprint(f"[green]💾 Research brief saved to:[/green] {save_path}")
        
    except Exception as e:
        rprint(f"[red]Error saving file: {e}[/red]")

# ===============================
# API FUNCTIONS
# ===============================

def call_research_api(topic: str, depth: int, audience: str, follow_up: bool, user_id: str, parent_session_id: str = None):
    """Call the FastAPI research endpoint"""
    try:
        url = f"{API_CONFIG['base_url']}/brief"
        payload = {
            "topic": topic,
            "depth": depth,
            "audience": audience,
            "follow_up": follow_up,
            "user_id": user_id,
            "parent_session_id": parent_session_id
        }
        
        rprint(f"[yellow]🌐 Calling API: {url}[/yellow]")
        
        response = requests.post(
            url,
            json=payload,
            timeout=API_CONFIG['timeout'],
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            return None, error_msg
            
    except requests.exceptions.Timeout:
        return None, "API request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to API. Is the server running?"
    except Exception as e:
        return None, f"API call failed: {str(e)}"

def get_user_history_api(user_id: str, limit: int = 10):
    """Get user history from API"""
    try:
        url = f"{API_CONFIG['base_url']}/users/{user_id}/history"
        params = {"limit": limit}
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('sessions', [])
        else:
            rprint(f"[yellow]Could not get history from API: {response.status_code}[/yellow]")
            return []
            
    except Exception as e:
        rprint(f"[yellow]API history request failed: {e}[/yellow]")
        return []

def test_api_connection():
    """Test if API server is running"""
    try:
        health_url = f"{API_CONFIG['base_url']}/health"
        response = requests.get(health_url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

# ===============================
# DISPLAY FUNCTIONS
# ===============================

def display_api_brief(api_response):
    """Display research brief from API response"""
    rprint(f"[bold cyan]Topic:[/bold cyan] {api_response.get('topic', 'Unknown')}")
    rprint(f"[bold cyan]Audience:[/bold cyan] {api_response.get('audience', 'Unknown')}")
    rprint(f"[bold cyan]Depth Level:[/bold cyan] {api_response.get('depth', 'Unknown')}/5")
    
    if api_response.get('is_follow_up'):
        rprint("[yellow]📎 This is a follow-up research building on previous work[/yellow]")
    
    rprint()
    
    # Thesis
    thesis = api_response.get('thesis')
    if thesis:
        rprint("[bold green]📝 Thesis:[/bold green]")
        rprint(Panel(thesis, border_style="green"))
    
    # Sections
    sections = api_response.get('sections', [])
    if sections:
        rprint("\n[bold blue]📚 Research Sections:[/bold blue]")
        for i, section in enumerate(sections, 1):
            heading = section.get('heading', f'Section {i}')
            content = section.get('content', 'No content available')
            rprint(f"\n[bold]{i}. {heading}[/bold]")
            rprint(content)
    
    # References
    references = api_response.get('references', [])
    if references:
        rprint(f"\n[bold magenta]🔗 References ({len(references)} sources):[/bold magenta]")
        for i, ref in enumerate(references, 1):
            title = ref.get('title', "Untitled Source")
            url = ref.get('url', "No URL")
            rprint(f"  {i}. {title}")
            rprint(f"     [dim]{url}[/dim]")

def display_api_user_history(user_history):
    """Display user history from API response"""
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
        # Handle API response format
        created_at = session.get('created_at', 'N/A')
        if created_at != 'N/A':
            try:
                date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except:
                date = 'N/A'
        else:
            date = 'N/A'
            
        topic = session.get('topic', 'Unknown Topic')
        topic = topic[:40] + '...' if len(topic) > 40 else topic
        
        session_type = "Follow-up" if session.get('is_follow_up', False) else "Initial"
        depth = str(session.get('depth', 'N/A'))
        
        table.add_row(date, topic, session_type, depth)
    
    rprint(table)

def display_context_summary(context_summary):
    """Display the context summary in a nice format"""
    rprint("\n[bold yellow]📋 Context from Previous Research[/bold yellow]")
    
    if isinstance(context_summary, dict):
        previous_topics = context_summary.get('previous_topics', [])
        if previous_topics:
            rprint("[blue]Previous Topics:[/blue]")
            for topic in previous_topics:
                rprint(f"  • {topic}")
        
        key_findings = context_summary.get('key_findings', [])
        if key_findings:
            rprint("\n[green]Key Findings to Build Upon:[/green]")
            for finding in key_findings[:5]:
                rprint(f"  • {finding}")
        
        knowledge_gaps = context_summary.get('knowledge_gaps', [])
        if knowledge_gaps:
            rprint("\n[red]Knowledge Gaps to Address:[/red]")
            for gap in knowledge_gaps:
                rprint(f"  • {gap}")
    else:
        rprint(f"[dim]{context_summary}[/dim]")
    
    rprint()

# ===============================
# DIRECT MODE FUNCTIONS (Fallback)
# ===============================

def show_user_history_direct(user_id: str):
    """Show user history in direct mode (fallback when API not available)"""
    try:
        # Try to import your existing modules
        from models import get_history_store
        
        history_store = get_history_store()
        user_history = history_store.get_user_history(user_id, limit=10)
        
        if not user_history:
            rprint("[dim]📚 No previous research history found.[/dim]")
            return user_history
        
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
            
            depth = 'N/A'
            try:
                if hasattr(session, 'final_brief') and session.final_brief:
                    brief_data = json.loads(session.final_brief) if isinstance(session.final_brief, str) else session.final_brief
                    depth = str(brief_data.get('depth', 'N/A'))
            except:
                depth = 'N/A'
            
            table.add_row(date, topic, session_type, depth)
        
        rprint(table)
        return user_history
        
    except ImportError:
        rprint("[yellow]⚠️ Direct mode not available - missing required modules[/yellow]")
        rprint("[dim]Please ensure your graph and models modules are available[/dim]")
        return []
    except Exception as e:
        rprint(f"[red]Error loading history: {e}[/red]")
        return []

def run_direct_mode(topic: str, depth: int, audience: str, follow_up: bool, user_id: str, parent_session_id: str = None):
    """Run research in direct mode (fallback when API not available)"""
    try:
        # Try to import your existing modules
        from graphs import graph
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create state
        state = {
            "topic": topic,
            "depth": depth,
            "audience": audience,
            "follow_up": follow_up,
            "user_id": user_id,
            "session_id": session_id,
            "parent_session_id": parent_session_id,
            "created_at": datetime.now().isoformat(),
            "session_type": "follow_up" if follow_up else "initial",
            "is_follow_up": follow_up
        }
        
        rprint(f"\n[cyan]🚀 Starting {'context-aware' if follow_up else 'new'} research...[/cyan]")
        rprint(f"[dim]Session ID: {session_id[:8]}... | User: {user_id[:8]}...[/dim]")
        
        # Execute graph
        rprint("[yellow]🔄 Executing research workflow...[/yellow]")
        compiled_graph = graph.compile()
        final_state = compiled_graph.invoke(state)
        
        return final_state, session_id
        
    except ImportError as e:
        rprint(f"[red]❌ Direct mode not available - missing modules: {e}[/red]")
        rprint("[dim]Please ensure your graph and models modules are available[/dim]")
        return None, None
    except Exception as e:
        rprint(f"[red]❌ Error in direct mode: {e}[/red]")
        return None, None

# ===============================
# MAIN CHATBOT FUNCTION
# ===============================

def chatbot():
    """Enhanced chatbot with API integration option"""
    rprint(Panel.fit("[bold green]Welcome to Context-Aware Research Brief Chatbot[/bold green]"))
    
    # Check if API mode is enabled and test connection
    if API_CONFIG["enabled"]:
        rprint("[dim]Running in API mode - requests will be sent to FastAPI server[/dim]")
        if test_api_connection():
            rprint("[green]✅ API server is running[/green]")
        else:
            rprint("[red]❌ Cannot reach API server[/red]")
            use_direct = Confirm.ask("Fall back to direct graph execution?", default=True)
            if use_direct:
                API_CONFIG["enabled"] = False
            else:
                return
    else:
        rprint("[dim]Running in direct mode - using local graph execution[/dim]")
    
    rprint("[dim]Now supports follow-up research building on previous conversations![/dim]\n")

    # Get or create user ID
    user_id = get_or_create_user_id()
    
    # Show user's research history
    user_history = []
    if API_CONFIG["enabled"]:
        user_history = get_user_history_api(user_id)
        if user_history:
            display_api_user_history(user_history)
    else:
        user_history = show_user_history_direct(user_id)
    
    # Gather inputs
    topic = input("\n📝 Enter research topic: ").strip() or "AI in Healthcare"
    
    # Ask about follow-up
    follow_up = False
    if user_history:
        rprint(f"\n[yellow]You have {len(user_history)} previous research sessions.[/yellow]")
        follow_up = Confirm.ask("Is this a follow-up to your previous research?", default=False)
        
        if follow_up:
            rprint("[green]✓ This research will build upon your previous work![/green]")
    
    # Other inputs
    depth_input = input("Enter research depth (1–5, default=2): ").strip()
    depth = int(depth_input) if depth_input.isdigit() and 1 <= int(depth_input) <= 5 else 2
    audience = input("Enter target audience (default='general'): ").strip() or "general"
    save_path = input("Optional: Enter path to save JSON (press Enter to skip): ").strip() or None

    # Determine parent session
    parent_session_id = None
    if follow_up and user_history:
        if API_CONFIG["enabled"]:
            parent_session_id = user_history[0].get('id') or user_history[0].get('session_id')
        else:
            parent_session_id = user_history[0].id if hasattr(user_history[0], 'id') else None

    try:
        if API_CONFIG["enabled"]:
            # ===== API MODE =====
            rprint(f"\n[cyan]🌐 Starting {'context-aware' if follow_up else 'new'} research via API...[/cyan]")
            
            api_response, error = call_research_api(
                topic=topic,
                depth=depth,
                audience=audience,
                follow_up=follow_up,
                user_id=user_id,
                parent_session_id=parent_session_id
            )
            
            if error:
                rprint(f"[red]❌ API Error: {error}[/red]")
                return
            
            # Display results
            session_id = api_response.get('session_id')
            rprint(Panel.fit(f"[bold cyan]📋 Research Brief: '{topic}'[/bold cyan]"))
            
            # Display context summary if available
            context_summary = api_response.get('context_summary')
            if follow_up and context_summary:
                display_context_summary(context_summary)
            
            # Display the brief
            display_api_brief(api_response)
            
            # Show execution summary
            rprint(f"\n[bold]📊 Execution Summary[/bold]")
            rprint(f"Session ID: [dim]{session_id}[/dim]")
            rprint(f"Type: {'Follow-up Research' if follow_up else 'Initial Research'}")
            rprint(f"User ID: [dim]{user_id[:8]}...[/dim]")
            rprint("[green]✅ Research completed successfully via API![/green]")
            
            # Save to file if requested
            if save_path:
                save_research_results(api_response, save_path, topic)
            
        else:
            # ===== DIRECT MODE =====
            final_state, session_id = run_direct_mode(
                topic=topic,
                depth=depth,
                audience=audience,
                follow_up=follow_up,
                user_id=user_id,
                parent_session_id=parent_session_id
            )
            
            if not final_state:
                rprint("[red]❌ Direct mode execution failed[/red]")
                return
            
            rprint(Panel.fit(f"[bold cyan]📋 Research Brief: '{topic}'[/bold cyan]"))
            
            # Handle final state display (you can implement this based on your existing code)
            rprint(f"\n[bold]📊 Execution Summary[/bold]")
            rprint(f"Session ID: [dim]{session_id}[/dim]")
            rprint(f"Type: {'Follow-up Research' if follow_up else 'Initial Research'}")
            rprint(f"User ID: [dim]{user_id[:8]}...[/dim]")
            rprint("[green]✅ Research completed successfully in direct mode![/green]")
            
    except Exception as e:
        rprint(f"[red]❌ Error: {str(e)}[/red]")

def ask_for_follow_up_research():
    """Ask user if they want to do follow-up research"""
    rprint("\n" + "="*50)
    
    continue_research = Confirm.ask("Would you like to conduct follow-up research on a related topic?", default=False)
    
    if continue_research:
        rprint("[green]🔄 Starting follow-up research...[/green]")
        rprint("="*50)
        chatbot()
    else:
        rprint("[blue]Thank you for using the Context-Aware Research Assistant![/blue]")
        rprint("[dim]Your research history has been saved for future sessions.[/dim]")

def main():
    """Main function with mode selection"""
    rprint(Panel.fit("[bold green]Context-Aware Research Assistant[/bold green]"))
    
    # Option to choose mode interactively
    use_api = Confirm.ask("Use API mode? (requires FastAPI server running)", default=False)
    if use_api:
        API_CONFIG["enabled"] = True
        custom_url = input("Enter API base URL (default: http://localhost:8000): ").strip()
        if custom_url:
            API_CONFIG["base_url"] = custom_url
    
    try:
        chatbot()
        ask_for_follow_up_research()
    except KeyboardInterrupt:
        rprint("\n[yellow]Session interrupted by user[/yellow]")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()