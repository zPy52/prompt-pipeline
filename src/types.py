from typing import TypeVar, Dict, Any, Optional, Generic, Union

from openai import Stream
from pydantic import BaseModel
from openai.types.responses import Response, ResponseStreamEvent

# Type variable for context models
ContextT = TypeVar("ContextT")
# Type variable for output models that must be Pydantic models
OutputT = TypeVar("OutputT", bound=BaseModel)

OpenAIResponse = Union[Response, Stream[ResponseStreamEvent]]

class PromptConfig(BaseModel):
  """Configuration for a single prompt in the pipeline"""
  model: str
  temperature: Optional[float] = None
  max_tokens: Optional[int] = None
  top_p: Optional[float] = None
  frequency_penalty: Optional[float] = None
  presence_penalty: Optional[float] = None
  stop: Optional[Union[str, list[str]]] = None
  
class PromptContext(Generic[ContextT]):
  """Wrapper for user-provided context with type safety"""
  def __init__(self, context: ContextT):
    self._context = context
    self._variables: Dict[str, Any] = {}
  
  @property
  def context(self) -> ContextT:
    return self._context
  
  def get_var(self, name: str) -> Any:
    return self._variables.get(name)
  
  def set_var(self, name: str, value: Any) -> None:
    self._variables[name] = value

class PipelineResponse(Generic[OutputT]):
  """Wrapper for pipeline responses with type safety"""
  def __init__(self, output: OutputT, raw_response: Any = None):
    self.output = output
    self.raw_response = raw_response
