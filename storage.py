# models/storage.py
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json
import logging
from typing import Optional, List, Dict, Any
import uuid
from contextlib import contextmanager

logger = logging.getLogger(__name__)

Base = declarative_base()


class UserSession(Base):
    __tablename__ = "user_sessions"
        
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)  # Added index for better performance
    topic = Column(String, nullable=False)
    final_brief = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_follow_up = Column(Boolean, default=False, nullable=False)
    parent_session_id = Column(String, nullable=True, index=True)  # Added index
    
    def __repr__(self):
        return f"<UserSession(id='{self.id}', user_id='{self.user_id}', topic='{self.topic}')>"
    
    def to_dict(self):
        """Convert to dictionary format for easier access"""
        return {
            'session_id': self.id,
            'user_id': self.user_id,
            'topic': self.topic,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_follow_up': self.is_follow_up,
            'parent_session_id': self.parent_session_id,
            'final_brief': self.get_final_brief_as_dict()
        }
    
    def get_final_brief_as_dict(self):
        """Safely parse final_brief JSON"""
        try:
            if isinstance(self.final_brief, str):
                return json.loads(self.final_brief)
            return self.final_brief
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Failed to parse final_brief for session {self.id}")
            return {}


class UserHistoryStore:
    def __init__(self, db_path: str = "user_history.db"):
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before use
            connect_args={"check_same_thread": False}  # Allow multi-threading
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
    
    @contextmanager
    def get_db_session(self):
        """Context manager for database sessions with proper cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def save_brief(self, user_id: str, session_id: str, topic: str,
                   final_brief: dict, is_follow_up: bool = False,
                   parent_session_id: Optional[str] = None) -> bool:
        """
        Save a completed research brief with enhanced error handling.
        
        Args:
            user_id: User identifier
            session_id: Unique session identifier
            topic: Research topic
            final_brief: Brief content as dictionary
            is_follow_up: Whether this is a follow-up session
            parent_session_id: ID of parent session if follow-up
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate inputs
            if not all([user_id, session_id, topic]):
                logger.error("Missing required parameters for save_brief")
                return False
            
            if not final_brief or not isinstance(final_brief, dict):
                logger.error("final_brief must be a non-empty dictionary")
                return False
            
            # Generate session_id if not provided or invalid
            if not session_id or session_id.strip() == "":
                session_id = str(uuid.uuid4())
                logger.info(f"Generated new session_id: {session_id}")
            
            with self.get_db_session() as session:
                # Check if session already exists
                existing = session.query(UserSession).filter(
                    UserSession.id == session_id
                ).first()
                
                # Convert final_brief to JSON with enhanced serialization
                try:
                    brief_json = json.dumps(final_brief, ensure_ascii=False, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Failed to serialize final_brief: {e}")
                    # Fallback: convert complex objects to strings
                    try:
                        sanitized_brief = self._sanitize_for_json(final_brief)
                        brief_json = json.dumps(sanitized_brief, ensure_ascii=False, indent=2)
                    except Exception as e2:
                        logger.error(f"Fallback serialization also failed: {e2}")
                        return False
                
                if existing:
                    logger.warning(f"Session {session_id} already exists, updating...")
                    existing.topic = topic
                    existing.final_brief = brief_json
                    existing.is_follow_up = is_follow_up
                    existing.parent_session_id = parent_session_id
                    existing.created_at = datetime.utcnow()  # Update timestamp
                else:
                    # Create new session
                    new_session = UserSession(
                        id=session_id,
                        user_id=user_id,
                        topic=topic,
                        final_brief=brief_json,
                        is_follow_up=is_follow_up,
                        parent_session_id=parent_session_id
                    )
                    session.add(new_session)
                
                logger.info(f"Successfully saved brief for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving brief: {e}")
            return False

    def _sanitize_for_json(self, obj):
        """Recursively sanitize objects for JSON serialization"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif hasattr(obj, 'model_dump'):
            try:
                return obj.model_dump(mode="json")
            except:
                return self._sanitize_for_json(obj.__dict__)
        elif hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    result[key] = self._sanitize_for_json(value)
            return result
        else:
            # Convert everything else to string
            return str(obj)

    def get_user_history(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent research history for a user, returning dictionaries instead of objects.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries
        """
        try:
            if not user_id:
                logger.error("user_id is required")
                return []
            
            with self.get_db_session() as session:
                results = (session.query(UserSession)
                          .filter(UserSession.user_id == user_id)
                          .order_by(UserSession.created_at.desc())
                          .limit(max(1, min(limit, 100)))  # Ensure reasonable limits
                          .all())
                
                # Convert to dictionaries for easier access
                history_dicts = []
                for result in results:
                    try:
                        history_dicts.append(result.to_dict())
                    except Exception as e:
                        logger.warning(f"Failed to convert session {result.id} to dict: {e}")
                        # Fallback: basic dictionary
                        history_dicts.append({
                            'session_id': result.id,
                            'user_id': result.user_id,
                            'topic': result.topic,
                            'created_at': result.created_at.isoformat() if result.created_at else None,
                            'is_follow_up': result.is_follow_up,
                            'parent_session_id': result.parent_session_id
                        })
                
                logger.info(f"Retrieved {len(history_dicts)} history items for user {user_id}")
                return history_dicts
                
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []

    def get_session_chain(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get full conversation chain for a session, returning dictionaries.
        
        Args:
            session_id: Starting session ID
            
        Returns:
            List of session dictionaries in chronological order
        """
        try:
            if not session_id:
                logger.error("session_id is required")
                return []
            
            sessions = []
            current_id = session_id
            visited_ids = set()  # Prevent infinite loops
            max_depth = 50  # Prevent runaway queries
            
            with self.get_db_session() as db_session:
                while current_id and current_id not in visited_ids and len(sessions) < max_depth:
                    visited_ids.add(current_id)
                    
                    session_obj = db_session.query(UserSession).filter(
                        UserSession.id == current_id
                    ).first()
                    
                    if session_obj:
                        try:
                            sessions.append(session_obj.to_dict())
                        except Exception as e:
                            logger.warning(f"Failed to convert session {current_id} to dict: {e}")
                            sessions.append({
                                'session_id': session_obj.id,
                                'user_id': session_obj.user_id,
                                'topic': session_obj.topic,
                                'created_at': session_obj.created_at.isoformat() if session_obj.created_at else None,
                                'is_follow_up': session_obj.is_follow_up,
                                'parent_session_id': session_obj.parent_session_id
                            })
                        current_id = session_obj.parent_session_id
                    else:
                        logger.warning(f"Session {current_id} not found, breaking chain")
                        break
                
                # Return in chronological order (oldest first)
                result = list(reversed(sessions))
                logger.info(f"Retrieved session chain with {len(result)} sessions")
                return result
                
        except Exception as e:
            logger.error(f"Error getting session chain: {e}")
            return []

    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific session by ID, returning a dictionary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session dictionary or None
        """
        try:
            if not session_id:
                return None
            
            with self.get_db_session() as session:
                result = session.query(UserSession).filter(
                    UserSession.id == session_id
                ).first()
                
                if result:
                    logger.info(f"Found session {session_id}")
                    try:
                        return result.to_dict()
                    except Exception as e:
                        logger.warning(f"Failed to convert session to dict: {e}")
                        return {
                            'session_id': result.id,
                            'user_id': result.user_id,
                            'topic': result.topic,
                            'created_at': result.created_at.isoformat() if result.created_at else None,
                            'is_follow_up': result.is_follow_up,
                            'parent_session_id': result.parent_session_id
                        }
                else:
                    logger.warning(f"Session {session_id} not found")
                    return None
                
        except Exception as e:
            logger.error(f"Error getting session by ID: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not session_id:
                return False
            
            with self.get_db_session() as session:
                result = session.query(UserSession).filter(
                    UserSession.id == session_id
                ).first()
                
                if result:
                    session.delete(result)
                    logger.info(f"Deleted session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False

    def get_brief_as_dict(self, session_id: str) -> Optional[Dict[Any, Any]]:
        """
        Get the final brief as a parsed dictionary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Parsed brief dictionary or None
        """
        try:
            session_dict = self.get_session_by_id(session_id)
            if not session_dict:
                return None
            
            return session_dict.get('final_brief', {})
            
        except Exception as e:
            logger.error(f"Error getting brief as dict: {e}")
            return None

    def update_session_topic(self, session_id: str, new_topic: str) -> bool:
        """
        Update the topic for an existing session.
        
        Args:
            session_id: Session identifier
            new_topic: New topic string
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not all([session_id, new_topic]):
                return False
            
            with self.get_db_session() as session:
                session_obj = session.query(UserSession).filter(
                    UserSession.id == session_id
                ).first()
                
                if session_obj:
                    session_obj.topic = new_topic
                    logger.info(f"Updated topic for session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for topic update")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating session topic: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.get_db_session() as session:
                total_sessions = session.query(UserSession).count()
                unique_users = session.query(UserSession.user_id).distinct().count()
                follow_up_sessions = session.query(UserSession).filter(
                    UserSession.is_follow_up == True
                ).count()
                
                return {
                    "total_sessions": total_sessions,
                    "unique_users": unique_users,
                    "follow_up_sessions": follow_up_sessions,
                    "database_path": self.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connections."""
        try:
            if hasattr(self, 'engine'):
                self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Utility functions
def create_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def validate_user_session_data(user_id: str, topic: str, final_brief: dict) -> bool:
    """
    Validate user session data before saving.
    
    Args:
        user_id: User identifier
        topic: Research topic
        final_brief: Brief content dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not user_id or not isinstance(user_id, str) or len(user_id.strip()) == 0:
        logger.error("Invalid user_id")
        return False
    
    if not topic or not isinstance(topic, str) or len(topic.strip()) == 0:
        logger.error("Invalid topic")
        return False
    
    if not final_brief or not isinstance(final_brief, dict):
        logger.error("Invalid final_brief")
        return False
    
    return True


# Factory function to get history store instance
_history_store_instance = None

def get_history_store(db_path: str = "user_history.db") -> UserHistoryStore:
    """Get or create a singleton UserHistoryStore instance"""
    global _history_store_instance
    if _history_store_instance is None:
        _history_store_instance = UserHistoryStore(db_path)
    return _history_store_instance


# Example usage and testing
if __name__ == "__main__":
    # Test the storage system
    store = UserHistoryStore("test_history.db")
    
    # Test data
    test_user_id = "user123"
    test_session_id = create_session_id()
    test_topic = "AI in Healthcare"
    test_brief = {
        "title": "AI in Healthcare Research Brief",
        "summary": "Comprehensive analysis of AI applications in healthcare...",
        "sections": ["Introduction", "Applications", "Challenges", "Future"],
        "sources": ["source1", "source2"]
    }
    
    # Test saving
    success = store.save_brief(
        user_id=test_user_id,
        session_id=test_session_id,
        topic=test_topic,
        final_brief=test_brief
    )
    
    print(f"Save successful: {success}")
    
    # Test retrieval
    history = store.get_user_history(test_user_id)
    print(f"Found {len(history)} history items")
    
    # Test statistics
    stats = store.get_stats()
    print(f"Database stats: {stats}")
    
    # Clean up
    store.close()