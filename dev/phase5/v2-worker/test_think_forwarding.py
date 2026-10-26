"""Offline regression for the one-line official RunPod Ollama handler patch."""

from __future__ import annotations

import ast
import argparse
import re
import unittest
from pathlib import Path


PATCH = Path(__file__).with_name("handler-think.patch")


def get_forwarded_fields(handler_path: Path | None = None) -> tuple[str, ...]:
    if handler_path is not None:
        source = handler_path.read_text(encoding="utf-8")
        matches = re.findall(r'^    for key in (\(.*\)):$', source, re.MULTILINE)
        if len(matches) != 1:
            raise AssertionError(f"Expected one handler forwarding allowlist, found {len(matches)}")
        return ast.literal_eval(matches[0])
    patch = PATCH.read_text(encoding="utf-8")
    added = re.search(r'^\+    for key in (\(.*\)):$', patch, re.MULTILINE)
    if added is None:
        raise AssertionError("Patch does not add a forwarding allowlist")
    return ast.literal_eval(added.group(1))


class ThinkForwardingTests(unittest.TestCase):
    handler_path: Path | None = None

    def test_patch_changes_only_the_supported_chat_field_allowlist(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")
        removed = re.findall(r'^-    for key in (\(.*\)):$', patch, re.MULTILINE)
        added = re.findall(r'^\+    for key in (\(.*\)):$', patch, re.MULTILINE)
        self.assertEqual(removed, ['("options", "format", "keep_alive", "tools", "system", "template")'])
        self.assertEqual(added, ['("options", "format", "keep_alive", "tools", "system", "template", "think")'])

    def test_false_is_forwarded_unchanged(self) -> None:
        forwarded_fields = get_forwarded_fields(self.handler_path)

        job_input = {"messages": [{"role": "user", "content": "test"}], "think": False}
        payload = {"model": "fixture", "messages": job_input["messages"]}
        for key in forwarded_fields:
            if key in job_input:
                payload[key] = job_input[key]

        self.assertIn("think", payload)
        self.assertIs(payload["think"], False)

    def test_absent_field_preserves_default_behavior(self) -> None:
        forwarded_fields = get_forwarded_fields(self.handler_path)
        job_input = {"messages": []}
        payload = {"model": "fixture", "messages": job_input["messages"]}
        for key in forwarded_fields:
            if key in job_input:
                payload[key] = job_input[key]
        self.assertNotIn("think", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handler", type=Path, help="test the patched official handler source")
    args, unittest_args = parser.parse_known_args()
    ThinkForwardingTests.handler_path = args.handler
    import sys

    sys.argv[1:] = unittest_args
    unittest.main(verbosity=2)
