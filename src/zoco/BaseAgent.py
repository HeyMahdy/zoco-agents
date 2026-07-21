
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union, AsyncGenerator, Dict
from dataclasses import dataclass, field
from openai import AsyncOpenAI

# ======================================================================
# 1. DATACLASSES 
# (Implementing the book's implied data structures)
# ======================================================================

@dataclass
class Message:
    content: str
    role: str

@dataclass
class UserMessage(Message):
    role: str = "user"
    source: str = "user"

@dataclass
class SystemMessage(Message):
    role: str = "system"

@dataclass
class AssistantMessage(Message):
    role: str = "assistant"

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
        return [{"role": msg.role, "content": msg.content} for msg in messages]

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

        return ChatCompletionResult(
            messages=AssistantMessage(
                content=response.choices[0].message.content
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
            *task_messages
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

        llm_messages = [
            SystemMessage(content=self.instructions),
            *self.context.messages,
            *task_messages
        ]

        completion_result = await self.model_client.create(llm_messages)
        assistant_message = completion_result.messages
        self.context.add_message(assistant_message)
        return assistant_message.content

        


