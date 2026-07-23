
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, AsyncGenerator, Dict, Callable
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from BaseTool import FunctionTool,BaseTool
import json
# ======================================================================
# 1. DATACLASSES 
# (Implementing the book's implied data structures)
# ======================================================================

@dataclass
class Message:
    content: str
    role: str

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class UserMessage(Message):
    role: str = "user"
    source: str = "user"

@dataclass
class SystemMessage(Message):
    role: str = "system"

@dataclass
class ToolMessage(Message):
    role: str = "tool"
    tool_call_id: str = ""

@dataclass
class AssistantMessage(Message):
    role: str = "assistant"
    tool_calls: Optional[List[ToolCall]] = None

@dataclass
class Usage:
    tokens_input: int
    tokens_output: int

@dataclass
class ChatCompletionResult:
    messages: AssistantMessage  
    usage: Usage
    model: str

@dataclass
class AgentContext:
    messages: List[Message] = field(default_factory=list)
    
    def add_message(self, message: Message):
        self.messages.append(message)






# Stubbing type hints that aren't defined in the snippet so it doesn't crash
AgentEvent = Any
AgentResponse = Any
BaseMemory = Any


# ======================================================================
# 2. THE BOOK'S INTERFACES
# ======================================================================

class BaseChatCompletionClient(ABC):
    @abstractmethod
    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatCompletionResult:
        pass

    @abstractmethod
    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionResult, None]:
        pass

class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        instructions: str,
        model_client: BaseChatCompletionClient,
        tools: Optional[List[Any]] = None,
        memory: Optional[BaseMemory] = None,  
        context: Optional[AgentContext] = None, 
        middleware: Optional[List[Any]] = None,
        max_iterations: Optional[int] = None
    ):
        self.name = name
        self.instructions = instructions
        self.model_client = model_client
        self.tools = tools if tools is not None else []
        self.context = context if context is not None else AgentContext() 
        self.memory = memory
        self.middleware = middleware if middleware is not None else []
        self.max_iterations = max_iterations
        
        self.current_iteration = 0

    @abstractmethod
    async def run(
        self, 
        task: Union[str, UserMessage, List[Message]]
    ) -> AgentResponse:
        pass

    @abstractmethod
    def run_stream(
        self, 
        task: Union[str, UserMessage, List[Message]]
    ) -> AsyncGenerator[Union[Message, AgentEvent], None]:
        pass


# ======================================================================
# 3. THE BOOK'S IMPLEMENTATIONS
# ======================================================================

class OpenAiChatCompletionClient(BaseChatCompletionClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__() 
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
    
    def _convert_messages_to_api_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        api_messages = []
        for msg in messages:
            # Base dictionary for all messages
            msg_dict = {"role": msg.role, "content": msg.content}
            
            # If it's an AssistantMessage, check for tool calls
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                # OpenAI needs tool calls in a very specific format
                formatted_tools = []
                for tc in msg.tool_calls:
                    formatted_tools.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    })
                msg_dict["tool_calls"] = formatted_tools
                
            # If it's a ToolMessage, add the tool_call_id
            elif isinstance(msg, ToolMessage):
                msg_dict["tool_call_id"] = msg.tool_call_id
                
            api_messages.append(msg_dict)
            
        return api_messages

    async def create(
            self,
            messages: List[Message],
            tools: Optional[List[Dict]] = None,
            **kwargs 
    ) -> ChatCompletionResult:
        
        api_messages = self._convert_messages_to_api_format(messages)

        # ---> THE FIX: Remove 'output_format' before passing to OpenAI <---
        kwargs.pop("output_format", None)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=tools,
            **kwargs
        )


        raw_message = response.choices[0].message
        
        # 2. Translate OpenAI's tool calls into your ToolCall dataclasses
        extracted_tool_calls = None
        if raw_message.tool_calls:
            extracted_tool_calls = []
            for tc in raw_message.tool_calls:
                extracted_tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments
                    )
                )

        return ChatCompletionResult(
            messages=AssistantMessage(
                content=response.choices[0].message.content,
                tool_calls=extracted_tool_calls
            ),
            usage=Usage(
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens
            ),
            model=response.model
        )

    # ---> ADDED: The missing abstract method <---
    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionResult, None]:
        
        api_messages = self._convert_messages_to_api_format(messages)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=tools,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content or ""
                yield ChatCompletionResult(
                    messages=AssistantMessage(content=content),
                    usage=Usage(tokens_input=0, tokens_output=0), # Stream chunks usually lack usage stats
                    model=chunk.model
                )

class Agent(BaseAgent):
    async def run_stream(
            self,
            task: Union[str, UserMessage, List[Message]]
    ) -> AsyncGenerator[Union[Message, AgentEvent], None]:
        
        if isinstance(task, str):
            task_messages = [UserMessage(content=task)]
        elif isinstance(task, Message):
            task_messages = [task]
        else:
            task_messages = task

        llm_messages = [
            SystemMessage(content=self.instructions),
            *self.context.messages,
        ]

        completion_result = await self.model_client.create(llm_messages)
        assistant_message = completion_result.messages

        yield assistant_message

        self.context.add_message(assistant_message)


    async def run(
            self,
            task: Union[str, UserMessage, List[Message]]

    )-> AgentResponse:
        
        if isinstance(task, str):
            task_messages = [UserMessage(content=task)]
        elif isinstance(task, Message):
            task_messages = [task]
        else:
            task_messages = task

        for msg in task_messages:
            self.context.add_message(msg)

        llm_messages = [
            SystemMessage(content=self.instructions),
            *self.context.messages,
        ]

        completion_result = await self.model_client.create(llm_messages, tools=self.llm_tools)
        assistant_message = completion_result.messages
        
        if assistant_message.tool_calls:
            # 1. ALWAYS add the assistant's request message to context first!
            self.context.add_message(assistant_message)

            processed_tools = self._process_tools(self.tools)

            for tc in assistant_message.tool_calls:
                target_tool = None
                for tool in processed_tools:
                    if tool.name == tc.name:
                        target_tool = tool
                        break

                if target_tool:
                    # Parse the string arguments into a dictionary
                    args_dict = json.loads(tc.arguments)
                    tool_result = await target_tool.execute(**args_dict)
                    content_str = str(tool_result.content)
                else:
                    content_str = "Error: Tool not found"

                # 2. Package the tool message with tc.id
                tool_msg = ToolMessage(
                    content=content_str,
                    role="tool",
                    tool_call_id=tc.id
                )
                self.context.add_message(tool_msg)

            # 3. Rebuild messages and call LLM again so it can give the final answer
            llm_messages = [
                SystemMessage(content=self.instructions),
                *self.context.messages,
                *task_messages
            ]
            
            final_result = await self.model_client.create(llm_messages, tools=self.llm_tools)
            final_message = final_result.messages
            self.context.add_message(final_message)
            return final_message.content

        else:
            self.context.add_message(assistant_message)
            return assistant_message.content

        
    def _process_tools(self, tools: List[Union[BaseTool, Callable]]) -> List[BaseTool]:
        processed = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed.append(tool)
            elif callable(tool):
                # Now this works! It wraps the callable in our new class
                processed.append(FunctionTool(tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return processed

    @property
    def llm_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Returns the tools in the exact JSON format the OpenAI client needs."""
        if not self.tools:
            return None
        return [tool.to_llm_format() for tool in self._process_tools(self.tools)]

    
            
        

        
