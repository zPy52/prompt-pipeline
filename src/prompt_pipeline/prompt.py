from openai import OpenAI
from typing import Type, Optional, Any, Dict, Generic

from .types import ContextT, OutputT, PromptConfig, PromptContext, PipelineResponse

class Prompt(Generic[ContextT, OutputT]):
  """
  A single prompt in the pipeline with type-safe context and output handling
  """
  def __init__(
    self,
    template: str,
    output_model: Type[OutputT],
    config: Optional[PromptConfig] = None,
    name: Optional[str] = None,
    depends_on: Optional[str] = None
  ):
    self.template = template
    self.output_model = output_model
    self.config = config or PromptConfig(model="gpt-4o-mini")
    self.name = name
    self.depends_on = depends_on
    self._format_args: Dict[str, Any] = {}
  
  def format(self, **kwargs: Any) -> 'Prompt[ContextT, OutputT]':
    """Add format arguments for the template"""
    self._format_args.update(kwargs)
    return self
  
  def model(self, model_name: str) -> 'Prompt[ContextT, OutputT]':
    """Set the model to use"""
    self.config.model = model_name
    return self
  
  def configure(self, **kwargs: Any) -> 'Prompt[ContextT, OutputT]':
    """Configure prompt parameters"""
    for key, value in kwargs.items():
      setattr(self.config, key, value)
    return self
  
  def _format_template(self, context: PromptContext[ContextT], prev_response: Optional[Any] = None) -> str:
    """Format the template with context and previous response"""
    format_args = {
      **self._format_args,
      "context": context.context,
      "prev": prev_response
    }
    
    # Add any variables stored in context
    for key, value in context._variables.items():
      if key not in format_args:
        format_args[key] = value
        
    return self.template.format(**format_args)

  def _modify_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively modify the JSON schema to set additionalProperties to false
    for all object types (Pydantic models)
    """
    # If this is an object type, set additionalProperties to false
    if schema.get("type") == "object":
      schema["additionalProperties"] = False
      
      # Recursively process properties if they exist
      properties = schema.get("properties", {})
      for k in properties:
        if isinstance(properties[k], dict):
          # Handle nested objects
          if properties[k].get("type") == "object":
            self._modify_schema(properties[k])
          # Handle arrays that might contain objects
          elif properties[k].get("type") == "array" and isinstance(properties[k].get("items"), dict):
            self._modify_schema(properties[k]["items"])
    
    return schema
  
  def execute(
    self,
    client: OpenAI,
    context: PromptContext[ContextT],
    prev_response: Optional[Any] = None
  ) -> PipelineResponse[OutputT]:
    """Execute the prompt and return a typed response"""
    formatted_prompt = self._format_template(context, prev_response)
    
    # Get the JSON schema and modify it
    schema = self.output_model.model_json_schema()
    modified_schema = self._modify_schema(schema)
    
    response = client.responses.create(
      model=self.config.model,
      input=formatted_prompt,
      temperature=self.config.temperature,
      max_output_tokens=self.config.max_tokens,
      top_p=self.config.top_p,
      text={
        "format": {
          "type": "json_schema",
          "name": "response",
          "strict": True,
          "schema": modified_schema
        }
      }
    )
    
    # Parse the response into the output model
    output = self.output_model.model_validate_json(response.output_text)
    
    return PipelineResponse(output=output, raw_response=response)
