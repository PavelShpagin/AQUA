"""
Utils package for GEC Judge

Contains utility modules for text processing and alignment.
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == 'align':
        from .alignment import align
        return align
    elif name == 'AlignmentResult':
        from .alignment import AlignmentResult
        return AlignmentResult
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['align', 'AlignmentResult'] 