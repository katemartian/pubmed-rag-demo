"""
Feature engineering utilities.
Deterministic transforms with type hints will go here.
"""
def dummy_feature(x: str) -> int:
    return len(x)