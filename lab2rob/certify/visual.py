"""Visual check: render scene cameras and ask a vision model what it sees.

Checks are GLOBAL scene properties — vessels in place, pipette still held,
liquid visible iff liquid ops ran. The chat vision backend cannot map well
names to pixels, so per-well visual grounding is out of scope until a
segmentation backend lands (and spills are bookkeeping, hence invisible:
no_spill stays predicate-only).

Backends: LLMBackend (live vision-model call, job "visual") and
FakeBackend (scripted verdicts for offline tests).
"""

from __future__ import annotations

import base64
import io
import json

CAMERAS = ["overhead", "front"]
WIDTH, HEIGHT = 640, 480

SYSTEM = """You inspect a rendered wet-lab robot scene (plate, reservoir block, two arms, pipette).
Reply with exactly one JSON object, no other text:
{"vessels_ok": <true|false>, "pipette_held": <true|false>, "liquid_visible": <true|false>, "detail": <short string>}
vessels_ok: plate and reservoir sit on the table, upright, nothing knocked over.
pipette_held: the right gripper holds the pipette (not dropped or missing).
liquid_visible: any liquid surface is visible in wells or troughs."""


def render_cameras(env, cameras=CAMERAS, width: int = WIDTH, height: int = HEIGHT) -> dict:
    """Camera name -> PNG bytes, via the env's MuJoCo renderer."""
    from PIL import Image

    images = {}
    for camera in cameras:
        frame = env.render(camera=camera, width=width, height=height)
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        images[camera] = buffer.getvalue()
    return images


def expectations(program) -> dict:
    """What the renders should show: liquid iff any liquid op was planned."""
    liquid = any(op["skill"] in ("aspirate", "dispense")
                 for step in program.steps if step.status == "planned"
                 for op in step.ops)
    return {"liquid_expected": liquid}


def _verdict(passed: bool, detail: str, **extra) -> dict:
    return {"passed": passed, "detail": detail, **extra}


class FakeBackend:
    """Scripted verdicts for offline tests; records what it was shown."""

    def __init__(self, verdicts: list):
        self.verdicts = list(verdicts)
        self.calls = []

    def check(self, images: dict, exp: dict) -> dict:
        self.calls.append({"cameras": sorted(images), "expectations": dict(exp)})
        if not self.verdicts:
            raise AssertionError("FakeBackend out of scripted verdicts")
        return dict(self.verdicts.pop(0))


class LLMBackend:
    """Live vision-model check (job 'visual'). One retry on bad JSON."""

    def __init__(self, client=None):
        self.client = client  # Lab2RobClient default; fake transport in tests
        self.calls = 0

    def _client(self):
        if self.client is None:
            from lab2rob.llm import Lab2RobClient
            self.client = Lab2RobClient()
        return self.client

    @staticmethod
    def _messages(images: dict, exp: dict, error: str = "") -> list:
        parts: list = [{"type": "text",
                        "text": ("Inspect these scene renders and reply with the JSON verdict only."
                                 + error)}]
        for camera in sorted(images):
            blob = base64.b64encode(images[camera]).decode("ascii")
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{blob}"}})
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": parts}]

    @staticmethod
    def _parse(text: str, exp: dict) -> dict:
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"not JSON: {e}") from None
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        for key in ("vessels_ok", "pipette_held", "liquid_visible"):
            if not isinstance(data.get(key), bool):
                raise ValueError(f"{key} must be a boolean")
        liquid_ok = data["liquid_visible"] == exp["liquid_expected"]
        passed = bool(data["vessels_ok"] and data["pipette_held"] and liquid_ok)
        return _verdict(passed, str(data.get("detail", "")),
                        vessels_ok=data["vessels_ok"],
                        pipette_held=data["pipette_held"],
                        liquid_visible=data["liquid_visible"])

    def check(self, images: dict, exp: dict) -> dict:
        error = ""
        for _ in range(2):
            self.calls += 1
            try:
                text = self._client().complete(
                    "visual", self._messages(images, exp, error)).text
                return self._parse(text, exp)
            except ValueError as e:
                error = f" Your last reply was invalid ({e}). Reply with valid JSON only."
        return _verdict(False, f"vision backend failed twice; {error.strip()}")


def check_visual(env, program, backend, cameras=CAMERAS,
                 width: int = WIDTH, height: int = HEIGHT) -> dict:
    """Render cameras and grade the scene against program expectations."""
    images = render_cameras(env, cameras, width, height)
    exp = expectations(program)
    verdict = backend.check(images, exp)
    return {"passed": bool(verdict.get("passed", False)),
            "detail": str(verdict.get("detail", "")),
            "cameras": sorted(images),
            "expectations": exp,
            "images": images,
            "backend_verdict": {k: v for k, v in verdict.items()
                                if k not in ("passed", "detail")}}
