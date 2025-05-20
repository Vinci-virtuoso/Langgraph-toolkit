from functools import lru_cache

import os
from app.workflows.rag_workflow import create_rag_graph


@lru_cache()
def get_cached_graph():
    return create_rag_graph()
