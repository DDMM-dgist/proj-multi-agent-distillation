"""Provider-neutral PydanticAI runtime adapter (additive; core stays unchanged).

This package adds a fourth agent-execution frontend alongside claude/codex/manual
WITHOUT touching the scientific workflow engine. The canonical contracts remain the
JSON Schemas in ``orchestration/schema/`` and the validators in
``orchestration/exchange.py`` + ``validation/``; the Pydantic models here are only a
typed *parsing* layer for structured LLM output. A parsed result is NEVER accepted on
Pydantic success alone — it is always re-checked by the existing ``validate_agent_response``
before the controller sees it (see ``driver.py``).

Nothing here is imported by the core packages. ``pydantic`` is an optional dependency
(``pip install -e .[pydantic-ai]``); ``pydantic_ai`` is imported lazily only by the real
provider runtime, so importing this package and running the mock runtime needs no API key.
"""
