"""RAG utilities for GEC judging.

Avoid side-effect imports at package import time. Import modules explicitly
where needed to prevent initializing heavy backends (e.g., ChromaDB) when RAG
is disabled.
"""

__all__ = []
