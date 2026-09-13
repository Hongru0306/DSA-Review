"""Unified LLM client: DeepSeek and OpenAI-compatible (Qwen vLLM).

Ports ``scripts/run_construction_review_v2.py`` DeepSeekAPI (sync urllib, JSON
mode, thinking disabled, retries / backoff, strips thinking blocks) and
``evidence_gated_generate.py`` ChatClient (Qwen ``enable_thinking=False``). Also
provides ``async chat(...)`` for offline graph / query-cache construction
(``graph.llm_builder``).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any


def normalize_chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


# ---- JSON parsing and deterministic repair (ported from run_construction_review_v2.py) ----


def repair_invalid_json_backslash_escapes(value: str) -> str:
    """Quote only the illegal backslash escapes inside JSON strings."""
    output: list[str] = []
    in_string = False
    index = 0
    hexadecimal = set("0123456789abcdefABCDEF")
    while index < len(value):
        character = value[index]
        if character == '"':
            in_string = not in_string
            output.append(character)
            index += 1
            continue
        if in_string and character == "\\":
            if index + 1 >= len(value):
                output.append("\\\\")
                index += 1
                continue
            escaped = value[index + 1]
            if escaped in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                output.extend((character, escaped))
                index += 2
                continue
            if (
                escaped == "u"
                and index + 5 < len(value)
                and all(digit in hexadecimal for digit in value[index + 2 : index + 6])
            ):
                output.append(value[index : index + 6])
                index += 6
                continue
            output.append("\\\\")
            index += 1
            continue
        output.append(character)
        index += 1
    return "".join(output)


def repair_missing_json_delimiter_commas(value: str, *, max_repairs: int = 64) -> str | None:
    """Insert missing JSON commas only where structurally unambiguous; return None on ambiguity."""
    candidate = str(value)
    repairs = 0
    while repairs < max_repairs:
        try:
            json.loads(candidate)
            return candidate if repairs else None
        except json.JSONDecodeError as caught_error:
            if caught_error.msg != "Expecting ',' delimiter":
                return None
            error_position = int(caught_error.pos)

        next_index = error_position
        while next_index < len(candidate) and candidate[next_index].isspace():
            next_index += 1
        previous_index = next_index - 1
        while previous_index >= 0 and candidate[previous_index].isspace():
            previous_index -= 1
        if previous_index < 0 or next_index >= len(candidate):
            return None

        stack: list[str] = []
        in_string = False
        escaped = False
        structurally_invalid = False
        for character in candidate[:next_index]:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "{[":
                stack.append(character)
            elif character in "}]":
                expected = "{" if character == "}" else "["
                if not stack or stack[-1] != expected:
                    structurally_invalid = True
                    break
                stack.pop()
        if structurally_invalid or in_string or not stack:
            return None

        previous = candidate[previous_index]
        prefix = candidate[: previous_index + 1]
        previous_is_value_end = (
            previous in '}]"0123456789'
            or prefix.endswith("true")
            or prefix.endswith("false")
            or prefix.endswith("null")
        )
        following = candidate[next_index]
        if stack[-1] == "{":
            following_is_valid = following == '"'
        else:
            following_is_valid = following in '{["-0123456789tfn'
        if not previous_is_value_end or not following_is_valid:
            return None

        candidate = candidate[:next_index] + "," + candidate[next_index:]
        repairs += 1
    return None


def audited_repaired_json_object(parsed: Any, marker: str) -> dict[str, Any]:
    if isinstance(parsed, dict):
        parsed["_transport_json_repair"] = marker
        return parsed
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return {"items": parsed, "_transport_json_repair": f"{marker}_wrapped_items"}
    raise TypeError("repaired response JSON is not an object or task-item array")


def parse_json_object_content(content: str) -> dict[str, Any]:
    """Parse a single model JSON object: strict JSON first, then deterministic repairs in order."""
    raw = str(content).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError("response JSON is not an object")
        return parsed
    except json.JSONDecodeError as initial_error:
        if raw.startswith("["):
            candidate = raw
        else:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0:
                raise ValueError("response did not contain a JSON object") from initial_error
            candidate = raw[start : end + 1] if end > start else raw[start:]

    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise TypeError("response JSON is not an object")
        return parsed
    except json.JSONDecodeError as strict_error:
        repaired_backslashes = repair_invalid_json_backslash_escapes(candidate)
        if repaired_backslashes != candidate:
            try:
                parsed = json.loads(repaired_backslashes)
                return audited_repaired_json_object(parsed, "invalid_backslash_escape")
            except json.JSONDecodeError:
                candidate = repaired_backslashes

        repaired_commas = repair_missing_json_delimiter_commas(candidate)
        if repaired_commas is not None:
            parsed = json.loads(repaired_commas)
            return audited_repaired_json_object(parsed, "missing_delimiter_comma")

        try:
            from json_repair import repair_json
        except ImportError:
            raise strict_error
        parsed = repair_json(candidate, return_objects=True)
        return audited_repaired_json_object(parsed, "json_repair")


# ---- Client ----


class LLMClient:
    """Thread-safe OpenAI-compatible client, Text / JSON modes + async chat."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 5,
        timeout: int = 120,
    ):
        if not api_key.strip():
            raise RuntimeError("api_key is required")
        self.api_key = api_key.strip()
        self.endpoint = normalize_chat_url(base_url)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._call_count = 0
        self._success_count = 0
        self._lock = threading.Lock()

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._call_count

    @property
    def successful_request_count(self) -> int:
        with self._lock:
            return self._success_count

    # ---- Request body ----

    def _body(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "qwen" in self.model.lower():
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            body["thinking"] = {"type": "disabled"}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _clean_content(self, content: str) -> str:
        content = re.sub(r" thinking.*? response", "", str(content), flags=re.S).strip()
        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S
            ).strip()
        return content

    # ---- Synchronous call (urllib) ----

    def _call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        last_error_detail = "unknown error"
        body = self._body(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "openai-python/1.0",
                    "Authorization": "Bearer " + self.api_key,
                },
                method="POST",
            )
            try:
                with self._lock:
                    self._call_count += 1
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = (
                    payload.get("choices", [{}])[0].get("message", {}).get("content")
                    or payload.get("output_text")
                    or ""
                )
                content = self._clean_content(content)
                with self._lock:
                    self._success_count += 1
                return content
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                json.JSONDecodeError,
            ) as exc:
                last_error_detail = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, urllib.error.HTTPError):
                    if exc.code in (401, 402, 403, 404):
                        last_error_detail += f"; response_body={exc.read().decode('utf-8', errors='replace')[:1000]}"
                        break
                if attempt < self.max_retries:
                    time.sleep(min(30.0, 1.5 * (2 ** min(attempt, 4)) + random.random()))
        raise RuntimeError(f"LLM request failed: {last_error_detail}")

    def call_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 900,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._call(messages, max_tokens=max_tokens, temperature=temperature, json_mode=False)

    def call_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1800,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for parse_attempt in range(1, self.max_retries + 1):
            content = self._call(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=True
            )
            try:
                return parse_json_object_content(content)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                if parse_attempt < self.max_retries:
                    time.sleep(min(8.0, 0.75 * (2**parse_attempt) + random.random()))
        raise ValueError(
            f"LLM returned invalid JSON after {self.max_retries} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )

    # ---- Asynchronous call (aiohttp, for offline cache construction) ----

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> str:
        import aiohttp

        body = self._body(messages, temperature=temperature, max_tokens=max_tokens, json_mode=False)
        last_error = "unknown"
        for attempt in range(4):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as session:
                    async with session.post(
                        self.endpoint,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    ) as resp:
                        text = await resp.text()
                        if resp.status == 200:
                            obj = json.loads(text)
                            return self._clean_content(
                                str(obj["choices"][0]["message"]["content"])
                            )
                        raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
            except Exception as exc:  # noqa: BLE001 - network retry
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 3:
                    break
                await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM chat failed: {last_error}")