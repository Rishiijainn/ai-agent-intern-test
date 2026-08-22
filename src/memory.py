import re
from typing import List, Dict, Any
from src.config import MAX_CONVERSATION_TURNS

class SessionMemory:
    """Manages multi-turn conversation history per session."""
    def __init__(self, max_turns: int = MAX_CONVERSATION_TURNS):
        self.max_turns = max_turns
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        
        # Context limits maintain karein
        if len(self.sessions[session_id]) > self.max_turns * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns * 2:]

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def contextualize_query(self, session_id: str, latest_query: str) -> str:
        """
        Follow-up questions ka context maintain karta hai (e.g. 'What about Canada?').
        """
        history = self.get_history(session_id)
        if not history:
            return latest_query

        prev_user_queries = [m["content"] for m in history if m["role"] == "user"]
        if not prev_user_queries:
            return latest_query

        last_query = prev_user_queries[-1]
        
        is_short_followup = len(latest_query.split()) <= 6 or any(
            w in latest_query.lower() for w in ["what about", "how about", "and canada", "when will it", "why", "where is it"]
        )

        if is_short_followup:
            return f"{last_query} {latest_query}"

        return latest_query