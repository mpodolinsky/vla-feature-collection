"""Backbone-feature identity checks, driven by a ``FeatureSpec`` (GR00T layer-16 blocks from
``serve_groot_ws.py --with-features``, or pi0.5 prefix block outputs from ``serve_pi05.py``).
Fails loudly on any shape / finiteness / token-split problem."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from constants import GROOT_SPEC, FeatureSpec
from records import AlignmentError, GrootFeatures

EXTRA_KEY = re.compile(r"^(base_image|wrist_image|language)@(.+)$")


def required_keys(spec: FeatureSpec) -> tuple[str, ...]:
    keys = ("base_image", "wrist_image", "language", "language_mask")
    return keys + (("state_features",) if spec.state_dim is not None else ())


def identify_groot_features(
    features: dict[str, Any], actions: np.ndarray, spec: FeatureSpec = GROOT_SPEC
) -> GrootFeatures:
    """Check one policy reply's features against ``spec`` and return them as fp16 arrays.

    Besides the primary ``base_image`` / ``wrist_image`` / ``language`` keys, any
    ``<name>@<tag>`` keys (extra captured layers) are validated and kept in ``extra``.
    """
    missing = [k for k in required_keys(spec) if k not in features]
    if missing:
        raise AlignmentError(
            f"backbone features missing {missing}. Start the server with --with-features."
        )
    base = np.asarray(features["base_image"])
    wrist = np.asarray(features["wrist_image"])
    language = np.asarray(features["language"])
    mask = np.asarray(features["language_mask"])
    state = np.asarray(features["state_features"]) if spec.state_dim is not None else None

    img_shape, lang_shape = (spec.img_tokens, spec.hidden), (spec.lang_max, spec.hidden)
    for name, arr, want in (("base_image", base, img_shape), ("wrist_image", wrist, img_shape),
                            ("language", language, lang_shape), ("language_mask", mask, (spec.lang_max,))):
        if arr.shape != want:
            raise AlignmentError(f"{name} shape {arr.shape} != {want}")
    if state is not None and state.shape != (spec.state_dim,):
        raise AlignmentError(f"state_features {state.shape} != ({spec.state_dim},)")

    extra: dict[str, np.ndarray] = {}
    for key, arr in features.items():
        m = EXTRA_KEY.match(key)
        if not m:
            continue
        arr = np.asarray(arr)
        want = lang_shape if m.group(1) == "language" else img_shape
        if arr.shape != want:
            raise AlignmentError(f"{key} shape {arr.shape} != {want}")
        extra[key] = arr.astype(np.float16)

    checked = [("base_image", base), ("wrist_image", wrist), ("language", language)] + list(extra.items())
    if state is not None:
        checked.append(("state_features", state))
    for name, arr in checked:
        if not np.isfinite(arr.astype(np.float32)).all():
            raise AlignmentError(f"{name} has NaN/Inf")
    if not bool(mask.any()):
        raise AlignmentError("language_mask is all-False (no instruction tokens)")
    real = language[mask.astype(bool)]
    if real.size and not np.any(real.astype(np.float32)):
        raise AlignmentError("language real region is all-zero")
    if np.allclose(base.astype(np.float32), wrist.astype(np.float32), atol=1e-4):
        raise AlignmentError("base_image and wrist_image are identical -- wrong token split")

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[0] < 1:
        raise AlignmentError(f"actions must be (horizon, dim), got {actions.shape}")
    if not np.isfinite(actions.astype(np.float32)).all():
        raise AlignmentError("actions have NaN/Inf")

    lang_len = features.get("language_len")
    lang_len = int(mask.sum()) if lang_len is None else int(lang_len)

    shapes = {
        "base_image": tuple(base.shape), "wrist_image": tuple(wrist.shape),
        "language": tuple(language.shape), "language_mask": tuple(mask.shape),
        "actions": tuple(actions.shape),
    }
    if state is not None:
        shapes["state_features"] = tuple(state.shape)
    shapes.update({k: tuple(v.shape) for k, v in extra.items()})

    return GrootFeatures(
        base_image=base.astype(np.float16),
        wrist_image=wrist.astype(np.float16),
        language=language.astype(np.float16),
        language_mask=mask.astype(bool),
        language_len=lang_len,
        state_features=None if state is None else state.astype(np.float16),
        extra=extra,
        source=spec.source,
        module=spec.module,
        shapes=shapes,
    )
