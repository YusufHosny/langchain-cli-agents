# ABOUTME: Unit tests for the shared chat base: message flattening, JSON extraction and the
# ABOUTME: prompt-enforced structured-output retry loop, over a scripted fake backend.
from pydantic import BaseModel, Field

from langchain_cli_agents.chat import ChatCLIBase, extract_json, split_messages
from langchain_cli_agents.core import CLIResult
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class _Scripted(ChatCLIBase):
  model: str = "fake"
  replies: list[str] = []
  seen: list[tuple[str, str | None]] = []

  @property
  def _llm_type(self) -> str:
    return "scripted"

  def _call(self, prompt: str, system: str | None) -> CLIResult:
    self.seen.append((prompt, system))
    return CLIResult(text=self.replies.pop(0), is_error=False,
                     usage={"input_tokens": 1, "output_tokens": 1})


class Answer(BaseModel):
  value: int = Field(..., description="the answer")


def test_split_prefixes_roles_only_for_multi_turn():
  assert split_messages([SystemMessage("s"), HumanMessage("h")]) == ("s", "h")
  _, conv = split_messages([SystemMessage("s"), HumanMessage("h"), AIMessage("a")])
  assert conv == "HUMAN: h\n\nAI: a"


def test_extract_json_handles_fences_and_prose():
  assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
  assert extract_json('sure! {"a": "}{", "b": [1]} done') == {"a": "}{", "b": [1]}
  assert extract_json("no json here") is None


def test_structured_output_retries_until_valid():
  llm = _Scripted(replies=["nope", '{"value": "x"}', '{"value": 42}'], seen=[])
  out = llm.with_structured_output(Answer).invoke([("system", "sys"), ("human", "q")])
  assert out == Answer(value=42)
  assert "JSON Schema" in llm.seen[0][1]
  assert "did not validate" in llm.seen[2][0]
  assert llm.usage_totals()["input_tokens"] == 3


def test_structured_output_gives_up_with_none():
  llm = _Scripted(replies=["a", "b", "c"], seen=[])
  assert llm.with_structured_output(Answer).invoke("q") is None
