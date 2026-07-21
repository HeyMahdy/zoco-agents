# Zoco Agents

> An experimental, open-source foundation for building agentic AI systems in Python.

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/HeyMahdy/zoco-agents)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Zoco Agents is a Python framework being built from scratch for creating AI agents with model clients, conversation context, streaming responses, tools, memory, and middleware.

The project is currently in its first public stage. The core abstractions are being explored and implemented, so the API is expected to change as the framework develops.

## Why Zoco Agents?

The goal is to provide a small, understandable foundation for agentic applications rather than hide the important pieces behind a large abstraction. Zoco is being designed around composable building blocks:

- agents with explicit instructions and execution lifecycles;
- interchangeable chat-completion clients;
- conversation context and message types;
- streaming responses;
- future support for tools, memory, middleware, and multi-agent workflows.

## Project status

**Current version: `0.1.0` — experimental**

Zoco Agents is not production-ready yet. It is suitable for experimentation, learning, prototyping, and contributing to the early design of the framework. Interfaces and behavior may change between versions, including breaking changes.

> **Warning:** Here be dragons. This project is under active development. Do not depend on the current API for production systems without pinning a version and reviewing changes carefully.

## Installation

The project currently uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
git clone https://github.com/HeyMahdy/zoco-agents.git
cd zoco-agents
uv sync
```

Set your OpenAI API key before running an agent:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Minimal example

The current prototype can be used with the built-in OpenAI chat-completion client:

```python
import asyncio
import os

from zoco.BaseAgent import Agent, OpenAiChatCompletionClient


async def main() -> None:
    client = OpenAiChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model_client=client,
    )

    response = await agent.run("Explain what an AI agent is in one paragraph.")
    print(response)


asyncio.run(main())
```

Because the package layout is still evolving, run examples from the repository with the source directory on the Python path:

```bash
PYTHONPATH=src uv run python your_example.py
```

## Roadmap

The roadmap is intentionally iterative and may change as the core design becomes clearer.

- [x] Define initial agent and message abstractions
- [x] Add an OpenAI chat-completion client
- [x] Add basic conversation context
- [ ] Add a stable tool-calling interface
- [ ] Add reliable streaming events
- [ ] Add memory and middleware implementations
- [ ] Add tests and continuous integration
- [ ] Improve package exports and public documentation
- [ ] Establish a stable `1.0.0` API

## Contributing

Contributions, feedback, issue reports, and design discussions are welcome. Before opening a pull request:

1. Read the existing code and documentation.
2. Open an issue for large API or architectural changes.
3. Keep changes focused and explain the reasoning behind them.
4. Add tests as the test suite becomes available.

```bash
git checkout -b feat/your-change
git add .
git commit -m "feat: describe your change"
git push origin feat/your-change
```

Then open a pull request on GitHub.

## Versioning

Zoco Agents follows semantic versioning as the public API matures. During the `0.x` phase, minor releases may contain breaking changes. Version `1.0.0` will mark the first stability commitment for the core API.

## License

Zoco Agents is released under the [MIT License](LICENSE).

## Acknowledgements

Zoco Agents is an independent project inspired by the broader open-source AI tooling ecosystem. The framework is being developed from first principles with an emphasis on clarity, composability, and practical experimentation.
