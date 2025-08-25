#!/usr/bin/env python3
"""
CLI Research Tool - Context-Aware Research Brief Client
Connects to your deployed API on Render for research brief generation
"""

import os
import json
import sqlite3
import requests
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

# Import PDF converter
from pdf_converter import generate_pdf_report, generate_research_paper, generate_brief_summary

console = Console()

# Configuration
API_BASE_URL = "https://context-based-research-brief-genrator.onrender.com"  # Replace with your actual Render URL
DB_FILE = "research_history.db"
USER_CONFIG_FILE = "user_config.json"

class ResearchDatabase:
    """Handle SQLite database operations for research history"""
    
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Research sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    audience TEXT NOT NULL,
                    is_follow_up BOOLEAN DEFAULT FALSE,
                    parent_session_id TEXT,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time REAL,
                    json_file_path TEXT,
                    pdf_file_path TEXT,
                    api_response TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            rprint("[green]Database initialized successfully[/green]")
    
    def create_user(self, username: str) -> str:
        """Create a new user and return user_id"""
        user_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (id, username) VALUES (?, ?)
                ''', (user_id, username))
                conn.commit()
                rprint(f"[green]User '{username}' created successfully[/green]")
                return user_id
            except sqlite3.IntegrityError:
                raise ValueError(f"Username '{username}' already exists")
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, created_at, last_active 
                FROM users WHERE username = ?
            ''', (username,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'created_at': row[2],
                    'last_active': row[3]
                }
        return None
    
    def update_last_active(self, user_id: str):
        """Update user's last active timestamp"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_active = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (user_id,))
            conn.commit()
    
    def save_research_session(self, session_data: Dict):
        """Save research session to database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO research_sessions 
                (session_id, user_id, username, topic, depth, audience, 
                 is_follow_up, parent_session_id, execution_time, 
                 json_file_path, pdf_file_path, api_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_data['session_id'],
                session_data['user_id'],
                session_data['username'],
                session_data['topic'],
                session_data['depth'],
                session_data['audience'],
                session_data['is_follow_up'],
                session_data.get('parent_session_id'),
                session_data.get('execution_time'),
                session_data.get('json_file_path'),
                session_data.get('pdf_file_path'),
                json.dumps(session_data.get('api_response', {}))
            ))
            conn.commit()
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's research history"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, topic, depth, audience, is_follow_up, 
                       created_at, execution_time, json_file_path, pdf_file_path
                FROM research_sessions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            return [{
                'session_id': row[0],
                'topic': row[1],
                'depth': row[2],
                'audience': row[3],
                'is_follow_up': bool(row[4]),
                'created_at': row[5],
                'execution_time': row[6],
                'json_file_path': row[7],
                'pdf_file_path': row[8]
            } for row in rows]
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT u.username, u.created_at, u.last_active,
                       COUNT(r.session_id) as research_count
                FROM users u
                LEFT JOIN research_sessions r ON u.id = r.user_id
                GROUP BY u.id
                ORDER BY u.last_active DESC
            ''')
            
            rows = cursor.fetchall()
            return [{
                'username': row[0],
                'created_at': row[1],
                'last_active': row[2],
                'research_count': row[3]
            } for row in rows]

class APIClient:
    """Handle API communication with the deployed research service"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Research-CLI-Tool/1.0'
        })
    
    def check_health(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def generate_research_brief(self, request_data: Dict) -> Dict:
        """Generate research brief via API"""
        try:
            response = self.session.post(
                f"{self.base_url}/brief",
                json=request_data,
                timeout=300  # 5 minutes timeout for research
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                raise Exception(error_msg)
                
        except requests.Timeout:
            raise Exception("Request timed out. Research might be taking longer than expected.")
        except requests.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
    
    def get_user_history_from_api(self, user_id: str) -> Dict:
        """Get user history from API"""
        try:
            response = self.session.get(f"{self.base_url}/users/{user_id}/history")
            if response.status_code == 200:
                return response.json()
            else:
                return {"sessions": [], "total_count": 0}
        except requests.RequestException:
            return {"sessions": [], "total_count": 0}

def save_json_output(data: Dict, filename: str) -> str:
    """Save research data to JSON file"""
    output_dir = Path("research_outputs")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    return str(filepath)

def display_research_brief(brief_data: Dict):
    """Display the research brief in a formatted way"""
    console.print(Panel.fit(f"[bold cyan]Research Brief: {brief_data.get('topic', 'Unknown')}[/bold cyan]"))
    
    # Basic info
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold blue")
    table.add_column("Value")
    
    table.add_row("Topic", brief_data.get('topic', 'N/A'))
    table.add_row("Audience", brief_data.get('audience', 'N/A'))
    table.add_row("Depth", str(brief_data.get('depth', 'N/A')))
    table.add_row("Follow-up", "Yes" if brief_data.get('is_follow_up') else "No")
    table.add_row("Execution Time", f"{brief_data.get('execution_time_seconds', 0):.2f}s")
    
    console.print(table)
    
    # Thesis
    if brief_data.get('thesis'):
        console.print(f"\n[bold green]Thesis:[/bold green]")
        console.print(Panel(brief_data['thesis'], border_style="green"))
    
    # Sections
    sections = brief_data.get('sections', [])
    if sections:
        console.print(f"\n[bold blue]Research Sections ({len(sections)}):[/bold blue]")
        for i, section in enumerate(sections, 1):
            heading = section.get('heading', f'Section {i}')
            content = section.get('content', 'No content')
            console.print(f"\n[bold]{i}. {heading}[/bold]")
            console.print(content[:200] + "..." if len(content) > 200 else content)
    
    # References
    references = brief_data.get('references', [])
    if references:
        console.print(f"\n[bold magenta]References ({len(references)} sources):[/bold magenta]")
        for i, ref in enumerate(references[:5], 1):  # Show first 5
            title = ref.get('title', 'Untitled')
            url = ref.get('url', 'No URL')
            console.print(f"  {i}. {title}")
            console.print(f"     [dim]{url}[/dim]")
        
        if len(references) > 5:
            console.print(f"     [dim]... and {len(references) - 5} more sources[/dim]")

def display_user_history(history: List[Dict], username: str):
    """Display user's research history"""
    if not history:
        console.print(f"[dim]No research history found for {username}[/dim]")
        return
    
    console.print(f"\n[bold blue]Research History for {username} ({len(history)} sessions)[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", style="dim", width=12)
    table.add_column("Topic", style="cyan", min_width=30)
    table.add_column("Type", justify="center", width=10)
    table.add_column("Depth", justify="center", width=5)
    table.add_column("Time", justify="right", width=8)
    
    for session in history:
        date = session['created_at'][:10]  # YYYY-MM-DD
        topic = session['topic'][:40] + '...' if len(session['topic']) > 40 else session['topic']
        session_type = "Follow-up" if session.get('is_follow_up') else "Initial"
        depth = str(session.get('depth', 'N/A'))
        exec_time = f"{session.get('execution_time', 0):.1f}s" if session.get('execution_time') else 'N/A'
        
        table.add_row(date, topic, session_type, depth, exec_time)
    
    console.print(table)

def get_or_create_user() -> tuple[str, str]:
    """Get existing user or create new one"""
    db = ResearchDatabase()
    
    # Check if user config exists
    if os.path.exists(USER_CONFIG_FILE):
        try:
            with open(USER_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            username = config.get('username')
            user_id = config.get('user_id')
            
            if username and user_id:
                use_existing = Confirm.ask(f"Continue as user '{username}'?", default=True)
                if use_existing:
                    db.update_last_active(user_id)
                    return user_id, username
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Show existing users
    users = db.get_all_users()
    if users:
        console.print("\n[bold blue]Existing Users:[/bold blue]")
        user_table = Table()
        user_table.add_column("Username", style="cyan")
        user_table.add_column("Research Count", justify="center")
        user_table.add_column("Last Active", style="dim")
        
        for user in users[:10]:  # Show first 10 users
            user_table.add_row(
                user['username'],
                str(user['research_count']),
                user['last_active'][:10]
            )
        console.print(user_table)
    
    # Get username
    username = Prompt.ask("\nEnter username (or create new)", default="researcher").strip()
    
    # Check if user exists
    user = db.get_user(username)
    if user:
        console.print(f"[green]Welcome back, {username}![/green]")
        user_id = user['id']
        db.update_last_active(user_id)
    else:
        # Create new user
        user_id = db.create_user(username)
        console.print(f"[green]New user '{username}' created![/green]")
    
    # Save user config
    with open(USER_CONFIG_FILE, 'w') as f:
        json.dump({'username': username, 'user_id': user_id}, f)
    
    return user_id, username

def main():
    """Main CLI function"""
    console.print(Panel.fit("[bold green]Context-Aware Research CLI Tool[/bold green]"))
    console.print("[dim]Connected to your deployed Render API[/dim]\n")
    
    # Initialize
    api_client = APIClient(API_BASE_URL)
    db = ResearchDatabase()
    
    # Check API health
    console.print("[yellow]Checking API connection...[/yellow]")
    if not api_client.check_health():
        console.print(f"[red]❌ Cannot connect to API at {API_BASE_URL}[/red]")
        console.print("[yellow]Please check your API URL and ensure the service is running[/yellow]")
        return
    
    console.print("[green]✅ API connection successful[/green]")
    
    try:
        while True:
            # Get user
            user_id, username = get_or_create_user()
            
            # Show recent history
            history = db.get_user_history(user_id, limit=5)
            if history:
                display_user_history(history, username)
            
            # Get research parameters
            console.print(f"\n[bold cyan]New Research Session for {username}[/bold cyan]")
            
            topic = Prompt.ask("Research topic", default="AI in Healthcare").strip()
            
            # Check for follow-up
            follow_up = False
            parent_session_id = None
            if history:
                follow_up = Confirm.ask(f"Build on previous research? (You have {len(history)} sessions)", default=False)
                if follow_up:
                    parent_session_id = history[0]['session_id']  # Use most recent
                    console.print(f"[green]Building on: {history[0]['topic']}[/green]")
            
            depth = int(Prompt.ask("Research depth (1-5)", default="2", choices=["1", "2", "3", "4", "5"]))
            audience = Prompt.ask("Target audience", default="general").strip()
            
            # Prepare API request
            request_data = {
                "topic": topic,
                "depth": depth,
                "audience": audience,
                "follow_up": follow_up,
                "user_id": user_id,
                "parent_session_id": parent_session_id
            }
            
            # Generate research brief
            console.print(f"\n[yellow]🔍 Generating research brief...[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing research request...", total=None)
                
                try:
                    response = api_client.generate_research_brief(request_data)
                    progress.update(task, description="Research completed!")
                    
                except Exception as e:
                    console.print(f"[red]❌ Error: {e}[/red]")
                    continue
            
            # Display results
            console.print("\n" + "="*60)
            display_research_brief(response)
            
            # Save files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in " -_").strip()[:30]
            
            # Save JSON
            json_filename = f"{username}_{safe_topic}_{timestamp}.json"
            json_path = save_json_output(response, json_filename)
            console.print(f"\n[green]💾 JSON saved to: {json_path}[/green]")
            
            # Ask for PDF format preference
            console.print("\n[bold cyan]PDF Generation Options:[/bold cyan]")
            console.print("1. Brief Summary (5-10 pages)")
            console.print("2. Full Research Paper (15-25 pages)")
            console.print("3. Both formats")
            console.print("4. Skip PDF generation")
            
            pdf_choice = Prompt.ask("Choose PDF format", choices=["1", "2", "3", "4"], default="2")
            
            pdf_paths = []
            try:
                if pdf_choice in ["1", "3"]:  # Brief summary
                    brief_filename = json_filename.replace('.json', '_brief.pdf')
                    brief_path = generate_brief_summary(response, f"research_outputs/{brief_filename}")
                    pdf_paths.append(brief_path)
                    console.print(f"[green]📄 Brief summary saved to: {brief_path}[/green]")
                
                if pdf_choice in ["2", "3"]:  # Full research paper
                    paper_filename = json_filename.replace('.json', '_research_paper.pdf')
                    paper_path = generate_research_paper(response, f"research_outputs/{paper_filename}")
                    pdf_paths.append(paper_path)
                    console.print(f"[green]📄 Research paper saved to: {paper_path}[/green]")
                
                pdf_path = pdf_paths[0] if pdf_paths else None
                
            except Exception as e:
                console.print(f"[yellow]⚠️ PDF generation failed: {e}[/yellow]")
                pdf_path = None
            
            # Save to database
            session_data = {
                'session_id': response['session_id'],
                'user_id': user_id,
                'username': username,
                'topic': topic,
                'depth': depth,
                'audience': audience,
                'is_follow_up': follow_up,
                'parent_session_id': parent_session_id,
                'execution_time': response.get('execution_time_seconds'),
                'json_file_path': json_path,
                'pdf_file_path': pdf_path,
                'api_response': response
            }
            
            db.save_research_session(session_data)
            console.print("[green]✅ Session saved to database[/green]")
            
            # Ask for another research
            console.print("\n" + "="*60)
            if not Confirm.ask("Conduct another research?", default=True):
                break
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
    
    console.print("[blue]Thank you for using the Research CLI Tool![/blue]")

if __name__ == "__main__":
    main()