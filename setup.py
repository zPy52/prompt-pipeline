from setuptools import setup, find_packages

setup(
  name="prompt_pipeline",
  version="0.1.0",
  packages=find_packages(),
  install_requires=[
    "openai>=1.0.0",
    "pydantic>=2.0.0",
  ],
  author="Antonio Peña Peña",
  description="A Python package to simplify type-safe LLM prompts chaining",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  python_requires=">=3.8",
)
