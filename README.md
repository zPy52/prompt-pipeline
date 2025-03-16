# Prompt Pipeline

A modern, type-safe prompt pipeline for LLM interactions. This package provides a clean, intuitive API for building prompt pipelines with full type safety and modern Python features.

## Features

- 🔒 Full type safety with generics and Pydantic models
- 🔗 Chain-style API for building pipelines
- 🎨 Decorator support for clean prompt definitions
- 📦 Context sharing between prompts
- 🔄 Automatic dependency handling between prompts
- ⚡ Modern Python features and best practices

## Installation

```bash
pip install prompt-pipeline
```

## Quick Start

Here's a simple example of using the pipeline:

```python
from pydantic import BaseModel
from prompt_pipeline import PromptPipeline

# Define your models
class TranslatorContext(BaseModel):
    source_text: str
    target_language: str

class ContextModel(BaseModel):
    context: str

class TranslationModel(BaseModel):
    translation: str

# Chain-style API
pipeline = (
    PromptPipeline[TranslatorContext, TranslationModel]()
    .add_prompt(
        "Analyze the context of this text: {context.source_text}",
        output_model=ContextModel,
        name="context"
    )
    .add_prompt(
        "Translate this text to {context.target_language} using the context: {prev.context}",
        output_model=TranslationModel,
        name="translate",
        depends_on="context"
    )
)

# Execute the pipeline
result = pipeline.execute(TranslatorContext(
    source_text="Hello, world!",
    target_language="Spanish"
))
```

## Decorator Style

You can also use decorators for a more declarative style:

```python
from prompt_pipeline import PromptPipeline, prompt

class TranslationPipeline(PromptPipeline[TranslatorContext, TranslationModel]):
    @prompt(output_model=ContextModel)
    def get_context(self, context: TranslatorContext) -> ContextModel:
        """
        Analyze the context of this text: {context.source_text}
        Consider the target language: {context.target_language}
        """
        pass
    
    @prompt(output_model=TranslationModel, depends_on="get_context")
    def translate(self, context: TranslatorContext) -> TranslationModel:
        """
        Translate this text to {context.target_language}
        Using the context: {prev.context}
        """
        pass

# Use the pipeline
pipeline = TranslationPipeline()
result = pipeline.execute(context)
```

## Advanced Features

### Configuration

Configure prompts with specific parameters:

```python
pipeline = (
    PromptPipeline()
    .add_prompt("Your prompt", output_model=YourModel)
    .model("gpt-4")
    .configure(
        temperature=0.7,
        max_tokens=100
    )
)
```

### Context Variables

Share variables between prompts:

```python
context = pipeline.context(
    text="Hello",
    language="Spanish"
)
context.set_var("style", "formal")
```

### Single Prompt Execution

Execute a single prompt without building a pipeline:

```python
result = pipeline.execute_single(
    "Translate {text} to {language}",
    output_model=TranslationModel
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
