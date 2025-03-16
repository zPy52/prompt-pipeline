from functools import wraps
from typing import Type, Optional, Callable, Any, Union

from pydantic import BaseModel
from openai.types import ChatModel

from .prompt import Prompt
from .types import PromptContext

def prompt(
  output_model: Optional[Type[BaseModel]] = None,
  *,
  model: Union[str, ChatModel] = "gpt-4o-mini",
  depends_on: Optional[str] = None,
  name: Optional[str] = None
):
  """
  Decorator to create a prompt from a method
  
  Example:
    @prompt(output_model=TranslationModel, model="gpt-4o-mini")
    def translate(self, wrapper: Optional[PromptContext] = None, prev: Optional[Any] = None) -> TranslationModel:
      if wrapper:
        # Do some processing with the context
        api_result = some_api_call(wrapper.context.text)
        
        # Store historical data in the context wrapper
        if not hasattr(wrapper.context, 'api_history'):
          wrapper.context.api_history = []
        wrapper.context.api_history.append(api_result)
        
        # Use previous response if available
        prev_analysis = prev.analysis if prev else "No previous analysis"
        
        return f'''
        Previous step analysis: {prev_analysis}
        Translate this text to {wrapper.context.target_language}
        Using additional context from API: {api_result}
        Historical API calls: {wrapper.context.api_history}
        '''
      else:
        return "Default template with no context"
  """
  def decorator(func: Callable[..., Any]):
    # Get the return type annotation as the output model if not provided
    nonlocal output_model
    if output_model is None:
      return_type = func.__annotations__.get('return')
      if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        output_model = return_type
    
    if output_model is None:
      raise ValueError(f"No output model specified for prompt {func.__name__}")
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      # Handle context
      context = None
      prev_response = None
      
      # Try to get context from args/kwargs
      if args:
        context = args[0]
      elif 'context' in kwargs:
        context = kwargs['context']
      # Try to get context from pipeline
      elif hasattr(self, '_context'):
        context = self._context
      
      # Get previous response if this prompt depends on another
      if depends_on and hasattr(self, '_responses') and depends_on in self._responses:
        prev_response = self._responses[depends_on].output
      
      # Wrap context in PromptContext if it's not None and not already wrapped
      if context is not None and not isinstance(context, PromptContext):
        context = PromptContext(context)
      
      # Execute the function to get the template
      template = func(self, context, prev_response)
      if not isinstance(template, str):
        raise ValueError(f"Prompt function {func.__name__} must return a string template")
      
      # Create the prompt
      prompt_obj = Prompt(
        template=template,
        output_model=output_model,
        name=name or func.__name__,
        depends_on=depends_on
      ).model(model)
      
      # Store the prompt in the pipeline
      if hasattr(self, '_prompts'):
        self._prompts.append(prompt_obj)
      
      return prompt_obj
    
    return wrapper
  
  return decorator
