"""
ai/gemini_client.py
====================
Google Gemini API client for Float Chat.

Responsibilities
----------------
1. Load API key from environment variable GEMINI_API_KEY.
2. Construct a prompt that injects RAG context and requests a JSON query plan.
3. Parse and return the structured plan dict.
4. Signal availability so callers can fall back gracefully when the key
   is absent or the call fails.

Security note
-------------
The JSON returned by Gemini is validated by core.query_validator before
any execution — no raw LLM output ever reaches the executor.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert oceanographic data assistant for the Float Chat system.
Your sole task is to convert a user's natural-language question into a
structured JSON query plan.

SCHEMA
------
{
  "operation":  string — one of: avg | min | max | sum | count | median | select | compare
  "variable":   string — one of: temperature | salinity | pressure | depth |
                         latitude | longitude | float_id | qc_flag
  "filters": {
    "region":       string | list[string]  — ocean region name(s), optional
    "depth_range":  {"min": number, "max": number}  — in metres, optional
    "date_range":   {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}  — optional
    "qc_flag":      int | list[int]  — Argo QC flag values, optional
  },
  "group_by":   list[string] — optional, values: region | year | month | qc_flag
  "sort_by":    {"column": string, "order": "asc"|"desc"}  — optional
  "limit":      int — max rows to return, optional (default 100)
}

RULES
-----
- Return ONLY valid JSON inside a ```json … ``` fence. No prose.
- Do not invent columns or operations outside the schema.
- If the question cannot be mapped, return:
  {"operation": "select", "variable": null, "filters": {}}
- Capitalise region names exactly as: North Atlantic, South Atlantic,
  North Pacific, South Pacific, Indian Ocean, Southern Ocean,
  Mediterranean Sea, Arctic Ocean, Caribbean Sea, Coral Sea.
"""


class GeminiClient:
    """
    Wraps the Google Generative AI SDK for Float Chat query planning.

    Parameters
    ----------
    model_name : str
        Gemini model to use (default: gemini-1.5-flash).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._model = None
        self._available = False
        self._init_error: Optional[str] = None
        self._setup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when the API key is present and SDK initialised."""
        return self._available

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def build_query_plan(
        self,
        question: str,
        rag_context: str = "",
    ) -> dict[str, Any]:
        """
        Call Gemini with *question* (and optional *rag_context*) and return
        a structured query plan dict.

        Raises RuntimeError if Gemini is unavailable.
        """
        if not self._available:
            raise RuntimeError(
                f"Gemini unavailable: {self._init_error or 'API key not set'}"
            )

        prompt = self._build_prompt(question, rag_context)
        response_text = self._call_api(prompt)
        plan = self._parse_response(response_text)
        plan["_mode"] = "gemini"
        return plan

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        if not self._api_key:
            self._init_error = "GEMINI_API_KEY environment variable not set."
            logger.info("Gemini: %s", self._init_error)
            return
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            self._available = True
            logger.info("Gemini client initialised with model '%s'.", self._model_name)
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)
            logger.warning("Gemini initialisation failed: %s", exc)

    def _build_prompt(self, question: str, rag_context: str) -> str:
        parts = []
        if rag_context:
            parts.append(f"RELEVANT SCIENTIFIC CONTEXT:\n{rag_context}\n")
        parts.append(f"USER QUESTION:\n{question}")
        return "\n".join(parts)

    def _call_api(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(prompt)  # type: ignore[union-attr]
            return response.text
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Extract the JSON block from the model response."""
        # Try ```json ... ``` fence first
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            # Fallback: try to find raw JSON object
            m2 = re.search(r"\{.*\}", text, re.DOTALL)
            if m2:
                json_str = m2.group(0)
            else:
                raise ValueError(f"No JSON found in Gemini response: {text!r}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from Gemini: {exc}\nRaw: {json_str}") from exc

