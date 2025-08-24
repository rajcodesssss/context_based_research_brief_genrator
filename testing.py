#!/usr/bin/env python3
"""
Simplified research client with database storage and follow-up functionality
"""

import requests
import json
import uuid
from datetime import datetime
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from typing import Dict, Any, List, Optional
import sys
import os

# Import your storage module
try:
    from storage import UserHistoryStore, create_session_id
except ImportError:
    rprint("[red]Error: Could not import storage module. Make sure models/storage.py is available.[/red]")
    sys.exit(1)

# ===============================
# USER MANAGEMENT WITH DATABASE
# ===============================

class UserManager:
    def __init__(self, db_path: str = "user_history.db"):
        self.storage = UserHistoryStore(db_path)
        self.current_user = None
    
    def get_or_create_user(self):
        """Get user information and store in database"""
        user_name = Prompt.ask("What should I call you?", default="Research User").strip()
        user_id = str(uuid.uuid4())
        
        self.current_user = {
            "name": user_name,
            "id": user_id,
            "created_at": datetime.now().isoformat()
        }
        
        rprint(f"[green]Hello {user_name}! Let's start your research.[/green]")
        return self.current_user
    
    def get_user_research_history(self, limit: int = 5):
        """Get user's research history from database"""
        if not self.current_user:
            return []
        
        return self.storage.get_user_history(self.current_user['id'], limit)
    
    def save_research_session(self, session_data: dict):
        """Save research session to database"""
        if not self.current_user:
            return False
        
        return self.storage.save_brief(
            user_id=self.current_user['id'],
            session_id=session_data['session_id'],
            topic=session_data['topic'],
            final_brief=session_data,
            is_follow_up=session_data.get('follow_up', False),
            parent_session_id=session_data.get('parent_session_id')
        )

# ===============================
# API FUNCTIONS
# ===============================

def call_research_api(api_url: str, topic: str, depth: int, audience: str, 
                     follow_up: bool, user_id: str):
    """Call the research API"""
    try:
        payload = {
            "topic": topic,
            "depth": depth,
            "audience": audience,
            "follow_up": follow_up,
            "user_id": user_id
        }
        
        rprint("[yellow]🔍 Conducting research...[/yellow]")
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=300,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            error_msg = f"API Error {response.status_code}: {response.text[:200]}"
            return None, error_msg
            
    except requests.exceptions.Timeout:
        return None, "Request timed out (5 minutes)"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to API. Check URL and network."
    except Exception as e:
        return None, f"API call failed: {str(e)}"

# ===============================
# DISPLAY FUNCTIONS
# ===============================

def display_research_result(api_response):
    """Display the final research result"""
    topic = api_response.get('topic', 'Unknown Topic')
    
    # Header
    rprint(Panel.fit(f"[bold cyan]Research Results: {topic}[/bold cyan]"))
    
    # Show if follow-up
    if api_response.get('is_follow_up') or api_response.get('follow_up'):
        rprint("[yellow]📚 This research builds on your previous work[/yellow]\n")
    
    # Main content - Thesis
    thesis = api_response.get('thesis')
    if thesis:
        rprint("[bold green]Key Finding:[/bold green]")
        rprint(Panel(thesis, border_style="green"))
        rprint()
    
    # Sections
    sections = api_response.get('sections', [])
    if sections:
        for i, section in enumerate(sections, 1):
            heading = section.get('heading', f'Section {i}')
            content = section.get('content', 'No content available')
            rprint(f"[bold blue]{i}. {heading}[/bold blue]")
            rprint(f"{content}\n")
    
    # References count
    references = api_response.get('references', [])
    if references:
        rprint(f"[dim]📖 Based on {len(references)} sources[/dim]")

def show_user_history(user_manager: UserManager):
    """Show user's research history"""
    history = user_manager.get_user_research_history()
    
    if not history:
        rprint("[dim]No previous research found.[/dim]")
        return False
    
    rprint(f"\n[blue]Your Recent Research ({len(history)} topics):[/blue]")
    for i, session in enumerate(history[:3], 1):  # Show only last 3
        topic = session.get('topic', 'Unknown')
        date = session.get('created_at', '')
        if date:
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00')).strftime('%m/%d')
            except:
                date = 'Recent'
        rprint(f"  {i}. {topic[:50]}{'...' if len(topic) > 50 else ''} [dim]({date})[/dim]")
    
    return True

# ===============================
# MAIN RESEARCH FLOW
# ===============================

def conduct_research():
    """Main research function"""
    rprint(Panel.fit("[bold green]🔬 AI Research Assistant[/bold green]"))
    
    # Initialize user manager
    user_manager = UserManager()
    
    # Get user information
    user = user_manager.get_or_create_user()
    
    # Check for previous research
    has_history = show_user_history(user_manager)
    
    # Ask about follow-up if user has history
    follow_up = False
    if has_history:
        follow_up = Confirm.ask("\n📖 Build on your previous research?", default=False)
    
    # Get API URL
    api_url ="https://context-based-research-brief-genrator.onrender.com/brief"
    
    # Get research parameters
    topic = Prompt.ask("Research topic").strip()
    
    depth_input = Prompt.ask("Research depth (1-5)", default="3").strip()
    depth = int(depth_input) if depth_input.isdigit() and 1 <= int(depth_input) <= 5 else 3
    
    audience = Prompt.ask("Target audience", default="general").strip() or "general"
    
    # Conduct research
    try:
        api_response, error = call_research_api(
            api_url=api_url,
            topic=topic,
            depth=depth,
            audience=audience,
            follow_up=follow_up,
            user_id=user['id']
        )
        
        if error:
            rprint(f"[red]❌ Error: {error}[/red]")
            return False
        
        # Display results
        rprint("\n" + "="*60)
        display_research_result(api_response)
        
        # Save to database
        session_data = {
            'session_id': api_response.get('session_id', create_session_id()),
            'topic': topic,
            'audience': audience,
            'depth': depth,
            'follow_up': follow_up,
            'user_name': user['name'],
            'user_id': user['id'],
            'created_at': datetime.now().isoformat(),
            **api_response  # Include all API response data
        }
        
        success = user_manager.save_research_session(session_data)
        if success:
            rprint(f"\n[green]✅ Research saved for {user['name']}[/green]")
        else:
            rprint(f"\n[yellow]⚠️  Could not save to database[/yellow]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Unexpected error: {str(e)}[/red]")
        return False

def ask_for_follow_up():
    """Ask if user wants to continue with more research"""
    rprint("\n" + "="*60)
    continue_research = Confirm.ask("🔄 Do more research?", default=True)
    return continue_research

# ===============================
# MAIN PROGRAM
# ===============================

def main():
    """Main program loop"""
    try:
        while True:
            success = conduct_research()
            
            if not success:
                retry = Confirm.ask("Try again?", default=True)
                if not retry:
                    break
                continue
            
            # Ask for follow-up
            if not ask_for_follow_up():
                break
        
        rprint("\n[blue]👋 Thanks for using AI Research Assistant![/blue]")
        
    except KeyboardInterrupt:
        rprint("\n[yellow]⏹️  Session ended[/yellow]")
    except Exception as e:
        rprint(f"[red]❌ Error: {e}[/red]")

if __name__ == "__main__":
    main()