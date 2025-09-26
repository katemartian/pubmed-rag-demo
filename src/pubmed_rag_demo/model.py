"""
Simple sklearn model with save/load stubs.
"""
def dummy_model_predict(x: list[str]) -> list[int]:
    return [len(s) for s in x]