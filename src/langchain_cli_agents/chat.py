# ABOUTME: Shared LangChain BaseChatModel for CLI harnesses: message flattening, usage totals and
# ABOUTME: prompt-enforced structured output (JSON schema in the system prompt, parse + validate).
from __future__ import annotations

import abc
import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
  AIMessage, BaseMessage, HumanMessage, SystemMessage, convert_to_messages,
)
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from langchain_cli_agents.core import CLIResult, SystemMode

_logger = logging.getLogger("langchain_cli_agents.chat")

_STRUCTURED_FMT = ("You MUST respond with ONLY a single JSON object that conforms to "
                   "this JSON Schema. No markdown, no code fences, no commentary.\n"
                   "JSON Schema:\n{schema}")


def to_messages(x: Any) -> list[BaseMessage]:
  if isinstance(x, PromptValue):
    return x.to_messages()
  if isinstance(x, BaseMessage):
    return [x]
  if isinstance(x, str):
    return [HumanMessage(content=x)]
  if isinstance(x, Sequence):
    return convert_to_messages(x)   # accepts ("role", "text") tuples and dicts too
  raise TypeError(f"cannot coerce {type(x)} to messages")


# headless CLIs take one prompt string, so a conversation is flattened into
# (system, transcript); role prefixes only once there is more than a system+user pair
def split_messages(messages: list[BaseMessage]) -> tuple[str, str]:
  sys_parts, conv_parts = [], []
  for m in messages:
    content = m.content if isinstance(m.content, str) else str(m.content)
    if isinstance(m, SystemMessage):
      sys_parts.append(content)
    else:
      role = getattr(m, "type", "human")
      conv_parts.append(f"{role.upper()}: {content}" if len(messages) > 2 else content)
  return "\n\n".join(sys_parts), "\n\n".join(conv_parts)


def extract_json(text: str) -> Any | None:
  t = text.strip()
  if t.startswith("```"):
    t = t.split("```", 2)[1] if t.count("```") >= 2 else t
    if t.startswith("json"):
      t = t[4:]
    t = t.strip("` \n")
  try:
    return json.loads(t)
  except json.JSONDecodeError:
    pass
  # fall back to the first balanced {...} object, string-aware
  start = t.find("{")
  if start == -1:
    return None
  depth, in_str, esc = 0, False, False
  for i in range(start, len(t)):
    c = t[i]
    if in_str:
      if esc:
        esc = False
      elif c == "\\":
        esc = True
      elif c == '"':
        in_str = False
    elif c == '"':
      in_str = True
    elif c == "{":
      depth += 1
    elif c == "}":
      depth -= 1
      if depth == 0:
        try:
          return json.loads(t[start:i + 1])
        except json.JSONDecodeError:
          return None
  return None


def _empty_usage() -> dict[str, float]:
  return {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0,
          "cost_usd": 0.0}


class ChatCLIBase(BaseChatModel, abc.ABC):
  model: str
  timeout: int = 600
  structured_retries: int = 3
  system_mode: SystemMode = "append"

  model_config = {"arbitrary_types_allowed": True}

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._usage_totals = _empty_usage()

  # one stateless single-turn completion, no tools
  @abc.abstractmethod
  def _call(self, prompt: str, system: str | None) -> CLIResult:
    raise NotImplementedError("_call should be implemented by subclasses")

  def usage_totals(self) -> dict:
    return dict(self._usage_totals)

  def reset_usage(self) -> None:
    self._usage_totals = _empty_usage()

  @staticmethod
  def _usage_metadata(res: CLIResult) -> UsageMetadata:
    input_total = res.total_input_tokens
    return UsageMetadata(
      input_tokens=input_total,
      output_tokens=res.output_tokens,
      total_tokens=input_total + res.output_tokens,
      input_token_details=InputTokenDetails(
        cache_read=res.cache_read_tokens, cache_creation=res.cache_creation_tokens,
      ),
    )

  def _record(self, res: CLIResult) -> None:
    self._usage_totals["input_tokens"] += res.total_input_tokens
    self._usage_totals["output_tokens"] += res.output_tokens
    self._usage_totals["cache_read"] += res.cache_read_tokens
    self._usage_totals["cache_creation"] += res.cache_creation_tokens
    self._usage_totals["cost_usd"] += res.cost_usd

  def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
    system, conv = split_messages(list(messages))
    res = self._call(conv, system or None)
    self._record(res)
    msg = AIMessage(
      content=res.text,
      usage_metadata=self._usage_metadata(res),
      response_metadata={"model_name": self.model, "cost_usd": res.cost_usd},
    )
    return ChatResult(generations=[ChatGeneration(message=msg)])

  def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Runnable:  # type: ignore[override]
    fmt = _STRUCTURED_FMT.format(schema=json.dumps(schema.model_json_schema()))
    model = self

    def _run(x: Any, config=None) -> BaseModel | None:
      system, conv = split_messages(to_messages(x))
      system = f"{system}\n\n{fmt}" if system else fmt
      for attempt in range(model.structured_retries):
        # through invoke so usage_metadata / callbacks fire
        ai = model.invoke([SystemMessage(content=system), HumanMessage(content=conv)],
                          config=config)
        obj = extract_json(str(ai.content))
        if obj is None:
          _logger.debug("[%s] no JSON found (try %d)", model._llm_type, attempt)
          continue
        try:
          return schema.model_validate(obj)
        except Exception as e:
          _logger.debug("[%s] validate failed (try %d): %s", model._llm_type, attempt, e)
          conv = (f"{conv}\n\nYour previous output did not validate: {e}. "
                  "Return corrected JSON only.")
      return None

    return RunnableLambda(_run)
