"""
prompt_pipeline - A modern, type-safe prompt pipeline for LLM interactions
"""

from .prompt import Prompt
from .decorators import prompt
from .pipeline import PromptPipeline
from .types import PromptConfig, PromptContext, PipelineResponse

__version__ = "1.0.0"
__all__ = [
  "PromptPipeline",
  "Prompt",
  "PromptConfig",
  "PromptContext",
  "PipelineResponse",
  "prompt",
]
