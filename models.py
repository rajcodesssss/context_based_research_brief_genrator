from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl, PositiveInt, model_validator

# -----------------------------
# Enhanced Pydantic Schemas for Context Awareness
# -----------------------------

class ContextSummary(BaseModel):
    """Summary of previous research context"""
    previous_topics: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    related_areas: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class ResearchRequest(BaseModel):
    """Enhanced request with context awareness"""
    topic: str
    depth: int = Field(default=2, ge=1, le=5)
    audience: str = "general"
    follow_up: bool = False
    user_id: str = "default_user"
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None

class ResearchPlanStep(BaseModel):
    order: PositiveInt
    objective: str
    method: str
    expected_output: str

class ResearchPlan(BaseModel):
    topic: str
    depth: int = Field(1, ge=1, le=5)
    steps: List[ResearchPlanStep]
    # New field for context-aware planning
    builds_on_previous: bool = False
    context_notes: Optional[str] = None

    @model_validator(mode="after")
    def ensure_steps_are_ordered(self):
        if [s.order for s in self.steps] != sorted([s.order for s in self.steps]):
            raise ValueError("ResearchPlan.steps must be ordered")
        return self

class SourceSummary(BaseModel):
    source_id: str
    url: Optional[HttpUrl] = None
    title: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    credibility_notes: Optional[str] = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            HttpUrl: str,
            datetime: lambda v: v.isoformat(),
        }


class FinalBriefReference(BaseModel):
    source_id: str
    url: Optional[HttpUrl] = None
    title: Optional[str] = None

class FinalBriefSection(BaseModel):
    heading: str
    content: str

class FinalBrief(BaseModel):
    topic: str
    audience: str = "general"
    depth: int = Field(1, ge=1, le=5)
    thesis: str
    sections: List[FinalBriefSection]
    references: List[FinalBriefReference]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    # New fields for context tracking
    session_id: Optional[str] = None
    is_follow_up: bool = False
    parent_session_id: Optional[str] = None

# -----------------------------
# User History Storage
# -----------------------------
import json
import sqlite3
from pathlib import Path
import uuid

def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling HttpUrl and datetime objects"""
    if hasattr(obj, 'model_dump'):
        # For Pydantic models, use model_dump with mode="json" 
        try:
            return obj.model_dump(mode="json")
        except Exception:
            # Fallback: convert manually
            return convert_to_serializable(obj.model_dump())
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, '__str__') and 'Url' in type(obj).__name__:
        # Handle HttpUrl objects
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def convert_to_serializable(obj):
    """Convert complex objects to JSON serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__str__') and 'Url' in type(obj).__name__:
        # Handle HttpUrl and similar URL objects
        return str(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'model_dump'):
        # Pydantic models
        try:
            return obj.model_dump(mode="json")
        except Exception:
            # Fallback to dict conversion
            return convert_to_serializable(obj.__dict__)
    elif hasattr(obj, '__dict__'):
        # Regular objects with __dict__
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = convert_to_serializable(value)
        return result
    else:
        # Last resort: convert to string
        return str(obj)

class UserHistoryStore:
    """SQLite-based storage for user research history with proper JSON serialization"""
    
    def __init__(self, db_path: str = "user_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                final_brief TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_follow_up BOOLEAN DEFAULT FALSE,
                parent_session_id TEXT,
                FOREIGN KEY (parent_session_id) REFERENCES user_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_brief(self, user_id: str, session_id: str, topic: str, 
                   final_brief: dict, is_follow_up: bool = False, 
                   parent_session_id: str = None):
        """Save a completed research brief with proper JSON serialization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert final_brief to JSON-serializable format
            serializable_brief = safe_json_serialize(final_brief)
            brief_json = json.dumps(serializable_brief, ensure_ascii=False)
            
            cursor.execute('''
                INSERT INTO user_sessions 
                (id, user_id, topic, final_brief, is_follow_up, parent_session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, user_id, topic, brief_json, 
                  is_follow_up, parent_session_id))
            
            conn.commit()
            print(f"💾 Saved brief for session {session_id}")
            
        except Exception as e:
            print(f"❌ Error saving brief: {e}")
            # Try alternative serialization
            try:
                fallback_brief = convert_to_serializable(final_brief)
                brief_json = json.dumps(fallback_brief, ensure_ascii=False, default=str)
                
                cursor.execute('''
                    INSERT INTO user_sessions 
                    (id, user_id, topic, final_brief, is_follow_up, parent_session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, user_id, topic, brief_json, 
                      is_follow_up, parent_session_id))
                
                conn.commit()
                print(f"💾 Saved brief with fallback serialization for session {session_id}")
                
            except Exception as fallback_error:
                print(f"❌ Fallback serialization also failed: {fallback_error}")
                raise
        finally:
            conn.close()
    
    def get_user_history(self, user_id: str, limit: int = 5):
        """Get recent research history for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, topic, final_brief, created_at, is_follow_up, parent_session_id
            FROM user_sessions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            try:
                final_brief_data = json.loads(row[2])
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse final_brief for session {row[0]}: {e}")
                final_brief_data = {"error": "Could not parse stored data"}
            
            # Convert to a simple object for easier access
            class HistorySession:
                def __init__(self, session_id, topic, final_brief, created_at, is_follow_up, parent_session_id):
                    self.id = session_id
                    self.topic = topic
                    self.final_brief = final_brief
                    self.created_at = datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at
                    self.is_follow_up = bool(is_follow_up)
                    self.parent_session_id = parent_session_id
            
            history.append(HistorySession(
                session_id=row[0],
                topic=row[1],
                final_brief=final_brief_data,
                created_at=row[3],
                is_follow_up=row[4],
                parent_session_id=row[5]
            ))
        
        return history
    
    def get_session_chain(self, session_id: str):
        """Get full conversation chain for a session"""
        sessions = []
        current_id = session_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Walk back through the chain
        while current_id:
            cursor.execute('''
                SELECT id, topic, final_brief, created_at, is_follow_up, parent_session_id
                FROM user_sessions WHERE id = ?
            ''', (current_id,))
            
            row = cursor.fetchone()
            if row:
                try:
                    final_brief_data = json.loads(row[2])
                except json.JSONDecodeError:
                    final_brief_data = {"error": "Could not parse stored data"}
                
                session_data = {
                    'session_id': row[0],
                    'topic': row[1],
                    'final_brief': final_brief_data,
                    'created_at': row[3],
                    'is_follow_up': row[4],
                    'parent_session_id': row[5]
                }
                sessions.append(session_data)
                current_id = row[5]  # parent_session_id
            else:
                break
        
        conn.close()
        return list(reversed(sessions))  # Chronological order

# Global instance
_history_store = None

def get_history_store():
    """Get the global history store instance"""
    global _history_store
    if _history_store is None:
        _history_store = UserHistoryStore()
    return _history_store

# -----------------------------
# Mock generators (updated)
# -----------------------------
def make_mock_plan(topic: str, depth: int, context_aware: bool = False) -> ResearchPlan:
    """Generate a mock research plan based on topic and depth"""
    steps = [
        ResearchPlanStep(order=1, objective="Clarify scope", method="planning", expected_output="Plan"),
        ResearchPlanStep(order=2, objective="Identify sources", method="search", expected_output="Source list"),
        ResearchPlanStep(order=3, objective="Summarize sources", method="summarization", expected_output="Summaries"),
        ResearchPlanStep(order=4, objective="Compile brief", method="synthesis", expected_output="Final brief"),
    ]
    
    # Add context-aware step for follow-ups
    if context_aware:
        steps.insert(0, ResearchPlanStep(
            order=0, 
            objective="Review previous research", 
            method="context_analysis", 
            expected_output="Context summary"
        ))
        # Renumber subsequent steps
        for i, step in enumerate(steps[1:], 1):
            step.order = i
    
    # Add extra step for higher depth
    if depth > 3:
        steps.append(
            ResearchPlanStep(order=len(steps)+1, objective="Deep analysis", method="analysis", expected_output="Detailed insights")
        )
    
    return ResearchPlan(
        topic=topic, 
        depth=depth, 
        steps=steps,
        builds_on_previous=context_aware,
        context_notes="Incorporates previous research context" if context_aware else None
    )

def make_mock_source_summaries(topic: str, depth: int) -> List[SourceSummary]:
    """Generate mock source summaries based on topic and depth"""
    n = min(5, max(2, depth + 1))
    summaries = []
    
    for i in range(1, n + 1):
        # Create safe URL by replacing spaces and special characters
        safe_topic = topic.replace(' ', '-').replace('&', 'and').lower()
        safe_topic = ''.join(c for c in safe_topic if c.isalnum() or c == '-')
        
        summary = SourceSummary(
            source_id=f"source_{i}",
            url=f"https://example.com/{safe_topic}/{i}",
            title=f"Example Source {i}: {topic}",
            key_points=[
                f"Key finding {i}.1 about {topic}",
                f"Key finding {i}.2 regarding implications"
            ],
            credibility_notes="Mock source for demonstration purposes",
        )
        summaries.append(summary)
    
    return summaries

def compile_final_brief(
    topic: str, 
    depth: int, 
    audience: str, 
    summaries: Optional[List[SourceSummary]],
    session_id: Optional[str] = None,
    is_follow_up: bool = False,
    parent_session_id: Optional[str] = None,
    context_summary: Optional[ContextSummary] = None
) -> FinalBrief:
    """Compile final research brief from source summaries with context awareness"""

    # Handle case where summaries might be None or empty
    if not summaries:
        summaries = []

    # Create sections based on available data
    sections: List[FinalBriefSection] = []

    # Add context section for follow-ups
    if is_follow_up and context_summary:
        context_content = f"This research builds upon previous work on: {', '.join(context_summary.previous_topics)}.\n\n"
        if context_summary.key_findings:
            context_content += f"Previous key findings: {'; '.join(context_summary.key_findings[:3])}.\n\n"
        if context_summary.knowledge_gaps:
            context_content += f"This research addresses gaps in: {'; '.join(context_summary.knowledge_gaps[:2])}."
        
        sections.append(FinalBriefSection(
            heading="Context from Previous Research",
            content=context_content
        ))

    # Introduction section
    intro_content = f"This research brief covers {topic} at depth level {depth}, tailored for {audience} audience."
    if is_follow_up:
        intro_content += " This is a follow-up investigation building on previous research."
    
    sections.append(FinalBriefSection(
        heading="Introduction",
        content=intro_content
    ))

    # Add key findings
    if summaries:
        key_findings_content = "Key findings from research:\n" + "\n".join([
            f"• {point}" for summary in summaries for point in summary.key_points
        ])
        sections.append(FinalBriefSection(
            heading="Key Findings",
            content=key_findings_content
        ))
    else:
        sections.append(FinalBriefSection(
            heading="Key Findings",
            content="No source summaries available yet."
        ))

    # Add methodology section for higher depth
    if depth >= 3:
        methodology_content = f"This brief was compiled using a {depth}-level depth analysis with {len(summaries)} sources."
        if is_follow_up:
            methodology_content += " The research methodology incorporated context from previous investigations."
        
        sections.append(FinalBriefSection(
            heading="Methodology",
            content=methodology_content
        ))

    # Always add conclusion
    conclusion_content = f"This brief provides {'comprehensive' if depth >= 3 else 'basic'} coverage of {topic}. "
    if is_follow_up:
        conclusion_content += "This follow-up research addresses gaps identified in previous investigations. "
    conclusion_content += "Further research may be needed for complete understanding."
    
    sections.append(FinalBriefSection(
        heading="Conclusion",
        content=conclusion_content
    ))

    # Create references from summaries
    references: List[FinalBriefReference] = [
        FinalBriefReference(
            source_id=summary.source_id,
            url=summary.url,
            title=summary.title
        ) for summary in summaries
    ]

    # Generate thesis with context awareness
    thesis = f"An analysis of {topic} reveals key insights and implications for the {audience} audience."
    if is_follow_up:
        thesis = f"Building on previous research, this analysis of {topic} provides additional insights and addresses identified knowledge gaps for the {audience} audience."

    return FinalBrief(
        topic=topic,
        audience=audience,
        depth=depth,
        thesis=thesis,
        sections=sections,
        references=references,
        session_id=session_id,
        is_follow_up=is_follow_up,
        parent_session_id=parent_session_id
    )

# -----------------------------
# Validation helpers
# -----------------------------
def validate_research_plan(plan: ResearchPlan) -> bool:
    """Validate that a research plan is well-formed"""
    try:
        # Basic validation - check if plan has required fields
        return len(plan.steps) > 0 and plan.topic.strip() != ""
    except Exception:
        return False

def validate_final_brief(brief: FinalBrief) -> bool:
    """Validate that a final brief is well-formed"""
    try:
        return len(brief.sections) > 0 and brief.thesis.strip() != ""
    except Exception:
        return False