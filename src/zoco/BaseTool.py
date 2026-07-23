import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class ToolResult:
    name: str
    content: Any 

class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Returns the JSON schema for the tool's parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Executes the tool with the provided arguments."""
        pass

    def to_llm_format(self) -> Dict[str, Any]: 
        """Converts the tool into the exact JSON format OpenAI expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters()
            }
        }


class FunctionTool(BaseTool):
    """Wraps a standard Python callable into an LLM-compatible Tool."""
    
    def __init__(self, func: Callable):
        self.func = func
        # 1. Automatically grab the function's name
        name = func.__name__
        
        # 2. Automatically grab the docstring for the description
        # (OpenAI uses this to know WHEN to use the tool)
        description = func.__doc__ or "No description provided."
        
        super().__init__(name=name, description=description)

    def parameters(self) -> Dict[str, Any]:
        """Uses Python's inspect module to build a JSON schema of the arguments."""
        sig = inspect.signature(self.func)
        
        properties = {}
        required = []

        # Loop through every argument the function takes
        for param_name, param in sig.parameters.items():
            # Skip 'self' if it's a class method
            if param_name == "self":
                continue
                
            # Map Python basic types to JSON schema types
            param_type = "string" # default fallback
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == float:
                param_type = "number"

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }

            # If the parameter has no default value, it is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            return ToolResult(name=self.name, content=str(result))
        except Exception as e:
            return ToolResult(name=self.name, content=f"Error executing tool: {str(e)}")



