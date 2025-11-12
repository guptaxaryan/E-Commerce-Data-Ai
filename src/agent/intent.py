"""Intent classification utilities for deciding between analysis and conversational reply."""
from __future__ import annotations
from typing import Dict
import re

import pandas as pd

ANALYTIC_KEYWORDS = {
    # Statistical operations
    "sum", "total", "average", "avg", "mean", "median", "min", "max", "count", "trend",
    "compare", "correlation", "distribution", "group", "aggregate", "plot", "chart",
    
    # Data operations
    "table", "rows", "columns", "filter", "top", "bottom", "percent", "rate", "growth",
    "show", "display", "list", "find", "search", "calculate", "analyze",
    
    # Business metrics
    "revenue", "sales", "profit", "cost", "price", "value", "amount", "quantity",
    "order", "customer", "product", "category", "seller", "payment",
    
    # Comparisons & rankings
    "highest", "lowest", "best", "worst", "most", "least", "rank", "ranking",
    "greater", "less", "more", "fewer",
    
    # Time-based
    "month", "quarter", "year", "week", "day", "period", "time", "date",
    "past", "last", "recent", "previous", "current", "since", "between",
    
    # Questions words (analytical context)
    "what", "which", "how many", "how much", "when", "where",
}

GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening)$",
    r"^(hi|hello|hey)[\s\!\.]*$",
    r"^(how are you|how's it going)\b",
    r"^(thanks|thank you)[\s\!\.]*$",
    r"^(bye|goodbye|see you)[\s\!\.]*$",
]


def classify_intent(question: str, dataframes: Dict[str, pd.DataFrame]) -> str:
    """Classify the user question as 'analysis' or 'chat'.

    Heuristics:
    - Pure greeting / pleasantry => chat
    - Presence of analytic keywords => analysis
    - Presence of table names or column names => analysis
    - Question words with data context => analysis
    - Otherwise: chat (but very rare now with expanded keywords)
    """
    q = question.lower().strip()
    
    # Greeting quick match - must be ONLY a greeting
    for pat in GREETING_PATTERNS:
        if re.match(pat, q):
            return "chat"
    
    # Check for analytical keywords FIRST (before table scan for speed)
    if any(kw in q for kw in ANALYTIC_KEYWORDS):
        return "analysis"

    # Check table or column names
    for tname, df in list(dataframes.items())[:12]:  # limit for speed
        if tname.lower() in q:
            return "analysis"
        for col in df.columns[:20]:  # limit columns scanned
            col_l = str(col).lower()
            if len(col_l) > 3 and col_l in q:
                return "analysis"

    # If question is longer than 5 words and not a greeting, assume analysis
    word_count = len(q.split())
    if word_count > 5:
        return "analysis"

    # Fallback to chat only for very short, non-analytical phrases
    return "chat"

__all__ = ["classify_intent"]
