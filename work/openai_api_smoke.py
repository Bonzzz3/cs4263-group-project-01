#!/usr/bin/env python3
"""Minimal OpenAI API connectivity and reasoning smoke test.

This is not part of the CS4263 evaluation. It only verifies that the local
SDK, API key, selected model, and reasoning setting work together.
"""

import os

from openai import OpenAI


model = os.environ.get("OPENAI_MODEL", "gpt-5.4")
effort = os.environ.get("OPENAI_REASONING_EFFORT", "high")

client = OpenAI()
response = client.responses.create(
    model=model,
    reasoning={"effort": effort},
    max_output_tokens=512,
    store=False,
    input=(
        "Answer this simple computer-science question. "
        "What is the time complexity of binary search on a sorted array? "
        "Return one short sentence."
    ),
)

usage = response.usage
print("model:", model)
print("reasoning_effort:", effort)
print("answer:", response.output_text)
print("input_tokens:", getattr(usage, "input_tokens", None))
print("output_tokens:", getattr(usage, "output_tokens", None))
details = getattr(usage, "output_tokens_details", None)
print("reasoning_tokens:", getattr(details, "reasoning_tokens", None))
