from typing import Dict
from sklearn.pipeline import Pipeline

from pipelines.processing import pipeline as processing_pipeline

@hook_impl
def register_pipelines(self) -> Dict[str, Pipeline]:
    """Register the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    p_processing = processing_pipeline.create_pipeline()

    return {"processing": p_processing}