"""Persistent response cache: hash(model + prompt + params) -> saved response.

Identical calls never pay twice: the key covers the model, the full message
list and any decoding parameters, so different models or temperatures never
collide. Corrupt or partial cache files are treated as misses.
"""

from __future__ import annotations

import hashlib
import json
import os


class LLMCache:
    def __init__(self, directory="runs/llm_cache"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def key(model, messages, **params):
        payload = {
            "model": model,
            "messages": messages,
            "params": {k: params[k] for k in sorted(params) if params[k] is not None},
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key):
        return os.path.join(self.directory, key + ".json")

    @staticmethod
    def _valid_response(data):
        if not isinstance(data, dict):
            return None
        resp = data.get("response")
        if not isinstance(resp, dict):
            return None
        if not isinstance(resp.get("text"), str):
            return None
        if not resp["text"].strip():
            return None  # empty replies are never served (see UI-04)
        for field in ("prompt_tokens", "completion_tokens"):
            if not isinstance(resp.get(field), int):
                return None
        return resp

    def get(self, model, messages, **params):
        try:
            with open(self._path(self.key(model, messages, **params)),
                      encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return self._valid_response(data)

    def put(self, model, messages, response, **params):
        key = self.key(model, messages, **params)
        record = {
            "key": key,
            "model": model,
            "messages": messages,
            "params": {k: params[k] for k in sorted(params) if params[k] is not None},
            "response": response,
        }
        path = self._path(key)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f)
        os.replace(tmp, path)
        return key
