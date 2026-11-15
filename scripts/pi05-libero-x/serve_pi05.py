#!/usr/bin/env python3
"""pi0.5 (LeRobot) + optional LoRA adapter behind an openpi-style websocket.

Runs in ``scripts/pi05-libero-x/.venv`` (LeRobot 0.7.0, Python 3.12). Same wire protocol as
``scripts/groot-libero-x/server/serve_groot_ws.py`` (msgpack-numpy, metadata dict on connect,
one reply per request), so the clients in the top-level venv talk to either policy.

Request (numpy, from ``eval_quick.build_obs``)::

    {"image": (H, W, 3) uint8, "wrist_image": (H, W, 3) uint8,
     "state": (8,) float32,    "prompt": str}

Reply: ``{"actions": (chunk_size, 7) float32}`` -- the full un-normalised chunk (50 steps);
the client decides how many to execute (the adapter was trained with ``n_action_steps=10``).

With ``--with-features`` the reply also carries hidden states of the PaliGemma Gemma-2B
backbone from the prefix pass, split by token type (fp16, padded language rows zeroed)::

    base_image (T, 2048)   wrist_image (T, 2048)   language (200, 2048)   language_mask (200,)
    language_len ()        T = 256 per camera, or 256 / pool**2 with --pool-image

``--layers`` picks the block outputs (0-based; 17 = last block, ``final`` = after the final norm).
The first entry is the primary (unsuffixed keys); further entries come back as ``base_image@L<k>``.
Block 17 and ``final`` are computed but never read by the action expert; block 16 is the last
block whose output feeds it (the expert's block 17 attends to K/V built from it).

    scripts/pi05-libero-x/.venv/bin/python scripts/pi05-libero-x/serve_pi05.py \\
        --adapter podolinsky/pi05-libero-x-level123-lora
    ... --no-adapter        # base lerobot/pi05_libero (control; uses the base's own processors)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import websockets.sync.server
from websockets.exceptions import ConnectionClosed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "groot-libero-x" / "server"))
import msgpack_numpy  # noqa: E402  (shared wire codec, vendored from openpi-client)

BASE_REPO = "lerobot/pi05_libero"
TOKENIZER_REPO = "google/paligemma-3b-pt-224"
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_adapter(adapter: str) -> str:
    """Local copy of the adapter with a ``tokenizer/`` dir.

    ``policy_preprocessor.json`` of ``podolinsky/pi05-libero-x-level123-lora`` declares a saved
    ``tokenizer`` artifact that the Hub repo does not contain, so LeRobot cannot build the
    preprocessor from the repo id. It is the stock PaliGemma tokenizer; save it next to the files.
    """
    from huggingface_hub import snapshot_download

    local = Path(adapter)
    if not local.is_dir():
        local = REPO_ROOT / "checkpoints" / adapter.split("/")[-1]
        snapshot_download(adapter, local_dir=str(local))
    if not (local / "tokenizer").is_dir():
        from transformers import AutoTokenizer

        logging.info("adapter has no tokenizer artifact; saving %s into %s", TOKENIZER_REPO, local / "tokenizer")
        AutoTokenizer.from_pretrained(TOKENIZER_REPO).save_pretrained(str(local / "tokenizer"))
    return str(local)


def load_policy(adapter: str | None, base: str, device: str, merge: bool):
    """Returns (policy, preprocessor, postprocessor)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if adapter:
        adapter = resolve_adapter(adapter)
    src = adapter or base
    cfg = PreTrainedConfig.from_pretrained(src)
    cfg.device = device
    cfg.compile_model = False  # no torch.compile warm-up for a quick server start
    policy = PI05Policy.from_pretrained(base, config=cfg)
    if adapter:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, adapter)
        if merge:
            policy = policy.merge_and_unload()  # plain PI05Policy with the LoRA folded in
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=src)
    return policy, pre, post


class FeatureTap:
    """Forward hooks on the backbone blocks; splits the prefix pass's hidden states by token type."""

    FP16_MAX = 65504.0

    def __init__(self, policy, layers: list[str], pool: int):
        cfg = policy.config
        lm = policy.model.paligemma_with_expert.paligemma.model.language_model
        self.layers, self.pool = layers, pool
        self.grid = int(cfg.image_resolution[0]) // 14        # SigLIP patch size 14 -> 16 per side
        self.n_img = self.grid * self.grid                      # 256 tokens per camera
        self.n_cams = len(cfg.image_features)                   # image, wrist, empty camera
        self.lang = int(cfg.tokenizer_max_length)
        if self.grid % pool:
            raise ValueError(f"--pool-image {pool} must divide the {self.grid}x{self.grid} patch grid")
        self.tokens_per_cam = (self.grid // pool) ** 2
        self.cap: dict[str, torch.Tensor] = {}
        self.n_clamped = 0
        n_blocks = len(lm.layers)
        for name in layers:
            if name == "final":
                mod = lm.norm
            else:
                k = int(name)
                if not 0 <= k < n_blocks:
                    raise ValueError(f"layer {k} out of range 0..{n_blocks - 1}")
                mod = lm.layers[k]
            mod.register_forward_hook(self._hook(name))

    def _hook(self, name: str):
        def fn(_module, _inputs, out):
            self.cap[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def clear(self) -> None:
        self.cap.clear()

    def _fp16(self, x: torch.Tensor) -> np.ndarray:
        x = x.float()
        if (x.abs() > self.FP16_MAX).any():
            self.n_clamped += 1
            logging.warning("fp16 overflow in a feature tensor (max %.0f); clamping", x.abs().max().item())
            x = x.clamp(-self.FP16_MAX, self.FP16_MAX)
        return x.half().cpu().numpy()

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:  # (n_img, C) -> (tokens_per_cam, C)
        if self.pool == 1:
            return tokens
        g, q, c = self.grid, self.pool, tokens.shape[-1]
        return tokens.reshape(g // q, q, g // q, q, c).float().mean(dim=(1, 3)).reshape(-1, c)

    def features(self, lang_mask: torch.Tensor) -> dict[str, np.ndarray]:
        mask = lang_mask[0].bool()
        out: dict[str, np.ndarray] = {
            "language_mask": mask.cpu().numpy(),
            "language_len": np.int32(int(mask.sum())),
        }
        for rank, name in enumerate(self.layers):
            h = self.cap[name][0]                                   # (prefix, 2048)
            want = self.n_cams * self.n_img + self.lang
            if h.shape[0] != want:
                raise RuntimeError(f"prefix has {h.shape[0]} tokens, expected {want} (token layout changed?)")
            sfx = "" if rank == 0 else f"@L{name}"
            out["base_image" + sfx] = self._fp16(self._pool(h[: self.n_img]))
            out["wrist_image" + sfx] = self._fp16(self._pool(h[self.n_img: 2 * self.n_img]))
            text = h[self.n_cams * self.n_img:] * mask[:, None]     # zero the padded slots
            out["language" + sfx] = self._fp16(text)
        return out


def to_chw(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


@torch.inference_mode()
def infer(policy, pre, post, obs: dict, tap: FeatureTap | None = None) -> dict:
    batch = {
        "observation.images.image": to_chw(obs["image"]),
        "observation.images.image2": to_chw(obs["wrist_image"]),
        "observation.state": torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)),
        "task": str(obs["prompt"]),
    }
    batch = pre(batch)
    if tap is not None:
        tap.clear()
    chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim), normalised
    chunk = post(chunk)
    result = {"actions": np.ascontiguousarray(chunk[0].float().cpu().numpy(), dtype=np.float32)}
    if tap is not None:
        result.update(tap.features(batch["observation.language.attention_mask"]))
    return result


class Server:
    def __init__(self, policy, pre, post, host: str, port: int, metadata: dict, tap: FeatureTap | None = None):
        self._p, self._pre, self._post, self._tap = policy, pre, post, tap
        self._host, self._port, self._meta = host, port, metadata

    def serve_forever(self) -> None:
        with websockets.sync.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            logging.info("pi0.5 websocket server ready on ws://%s:%d", self._host, self._port)
            server.serve_forever()

    def _handler(self, ws) -> None:
        packer = msgpack_numpy.Packer()
        ws.send(packer.pack(self._meta))
        logging.info("client connected: %s", ws.remote_address)
        while True:
            try:
                msg = ws.recv()
            except ConnectionClosed:
                logging.info("client disconnected")
                return
            try:
                result = infer(self._p, self._pre, self._post, msgpack_numpy.unpackb(msg), self._tap)
                ws.send(packer.pack(result))
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                logging.error("inference error:\n%s", tb)
                ws.send(tb)  # a str reply makes the openpi client raise RuntimeError


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="podolinsky/pi05-libero-x-level123-lora",
                   help="PEFT LoRA adapter (hub id or local dir); ignored with --no-adapter")
    p.add_argument("--no-adapter", action="store_true", help="serve the base model only (control)")
    p.add_argument("--base", default=BASE_REPO)
    p.add_argument("--no-merge", action="store_true", help="keep the LoRA unmerged (slower)")
    p.add_argument("--with-features", action="store_true", help="also return backbone hidden states")
    p.add_argument("--layers", default="16",
                   help="comma list of backbone block outputs to return, 0-based (17 = last block, "
                        "16 = last one the action expert reads) or 'final' (after the final norm); first = primary")
    p.add_argument("--pool-image", type=int, default=1,
                   help="average-pool each camera's 16x16 token grid by this factor (2 -> 64 tokens)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    adapter = None if args.no_adapter else args.adapter
    logging.info("loading base=%s adapter=%s", args.base, adapter)
    policy, pre, post = load_policy(adapter, args.base, args.device, merge=not args.no_merge)
    cfg = policy.config
    logging.info("policy loaded (chunk_size=%d, n_action_steps=%d)", cfg.chunk_size, cfg.n_action_steps)
    tap, feat_meta = None, {"with_features": False}
    if args.with_features:
        layers = [s.strip() for s in args.layers.split(",") if s.strip()]
        tap = FeatureTap(policy, layers, args.pool_image)
        feat_meta = {"with_features": True, "layers": layers, "pool_image": args.pool_image,
                     "img_tokens": tap.tokens_per_cam, "hidden": 2048, "lang_slots": tap.lang}
        logging.info("feature capture: %s", feat_meta)
    meta = {**feat_meta, "policy": "pi05-lerobot", "base": args.base, "adapter": adapter,
            "chunk_size": int(cfg.chunk_size), "n_action_steps": int(cfg.n_action_steps),
            "action_order": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}
    Server(policy, pre, post, args.host, args.port, meta, tap).serve_forever()


if __name__ == "__main__":
    main()
