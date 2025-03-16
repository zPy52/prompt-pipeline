from typing import Generic, List, Dict, Optional, Type, Any

from openai import OpenAI
from pydantic import BaseModel

from .types import ContextT, OutputT, PromptContext, PipelineResponse, PromptConfig
from .prompt import Prompt

class PromptPipeline(Generic[ContextT, OutputT]):
  """
  A type-safe pipeline for executing a sequence of prompts
  
  Example:
    pipeline = (
      PromptPipeline[Context, TranslationModel]()
      .add_prompt("Get context", output_model=ContextModel)
      .format("Analyze {text} for translation")
      .add_prompt("Translate", output_model=TranslationModel)
      .format("Translate using context: {prev.context}")
      .build()
    )
    
    result = pipeline.execute(context)
  """
  def __init__(self, client: Optional[OpenAI] = None):
    self.client = client or OpenAI()
    self.clear()
  
  def clear(
      self, 
      reset_prompts: bool = True, 
      reset_context: bool = True, 
      reset_responses: bool = True) -> None:
    if reset_prompts:
      self._prompts: List[Prompt] = []
    if reset_context:
      self._context: Optional[PromptContext] = None
    if reset_responses:
      self._responses: Dict[str, PipelineResponse] = {}

  def context(self, **kwargs: Any) -> 'PromptPipeline[ContextT, OutputT]':
    """Set the context for the pipeline"""
    self._context = PromptContext(kwargs)
    return self
  
  def add_prompt(
    self,
    template: str,
    *,
    output_model: Type[BaseModel],
    name: Optional[str] = None,
    depends_on: Optional[str] = None,
    config: Optional[PromptConfig] = None
  ) -> 'PromptPipeline[ContextT, OutputT]':
    """Add a prompt to the pipeline"""
    prompt = Prompt(
      template=template,
      output_model=output_model,
      name=name,
      depends_on=depends_on,
      config=config
    )
    self._prompts.append(prompt)
    self._current_prompt = prompt
    return self
  
  def format(self, **kwargs: Any) -> 'PromptPipeline[ContextT, OutputT]':
    """Format the current prompt template"""
    if self._current_prompt is None:
      raise ValueError("No prompt to format")
    self._current_prompt.format(**kwargs)
    return self
  
  def configure(self, **kwargs: Any) -> 'PromptPipeline[ContextT, OutputT]':
    """Configure the current prompt"""
    if self._current_prompt is None:
      raise ValueError("No prompt to configure")
    self._current_prompt.configure(**kwargs)
    return self
  
  def model(self, model_name: str) -> 'PromptPipeline[ContextT, OutputT]':
    """Set the model for the current prompt"""
    if self._current_prompt is None:
      raise ValueError("No prompt to set model for")
    self._current_prompt.model(model_name)
    return self
  
  def execute(self, context: Optional[ContextT] = None) -> OutputT:
    """Execute the pipeline and return the final output"""
    if context is not None:
      self._context = PromptContext(context)
    elif self._context is None:
      raise ValueError("No context provided")
    
    for prompt in self._prompts:
      # Get the previous response if this prompt depends on another
      prev_response = None
      if prompt.depends_on and prompt.depends_on in self._responses:
        prev_response = self._responses[prompt.depends_on].output
      
      # Execute the prompt
      response = prompt.execute(
        client=self.client,
        context=self._context,
        prev_response=prev_response
      )
      
      # Store the response
      if prompt.name:
        self._responses[prompt.name] = response
    
    # Return the last response's output
    if not self._prompts:
      raise ValueError("No prompts in pipeline")
    
    output = (
      self._responses[self._prompts[-1].name].output 
      if self._prompts[-1].name 
      else self._prompts[-1].output
    )

    return output
