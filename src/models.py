from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Citation(BaseModel):
    filename: str
    heading: str
    content_snippet: str

class SanitizedOrder(BaseModel):
    order_id: str
    status: str
    items_summary: List[str] = Field(default_factory=list)
    shipping_method: Optional[str] = None
    carrier: Optional[str] = None
    estimated_delivery: Optional[str] = None
    return_status: Optional[str] = None
    notes_for_customer: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    human_handoff_recommended: bool = False
    tool_called: Optional[str] = None
    sanitized_tool_args: Optional[Dict[str, Any]] = None
    debug_trace: Optional[Dict[str, Any]] = None