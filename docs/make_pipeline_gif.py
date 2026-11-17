#!/usr/bin/env python3
"""Render docs/pipeline.gif: a policy-agnostic overview of the collect -> record -> ship pipeline.

The simulator panel plays real frames from a shipped episode of the public dataset
(`podolinsky/pi05-libero-x-level1-features`); the output panel lists the keys and shapes pi0.5 produces
(see scripts/pi05-libero-x/DATASET_CARD.md). Needs ffmpeg and matplotlib.

    python docs/make_pipeline_gif.py [--out docs/pipeline.gif] [--still docs/pipeline_still.png]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ID = "podolinsky/pi05-libero-x-level1-features"
EPISODE = "LEVEL1/00_EXTENSION_KITCHEN_SCENE11_turn_on_the_flat_stove/ep000"

INK, GREY, LINE, BG = "#1f2328", "#6e7781", "#d0d7de", "#ffffff"
BLUE, ORANGE, SLATE, GREEN = "#2563eb", "#e07b00", "#57606a", "#1a7f37"
TINT = {BLUE: "#eaf1ff", ORANGE: "#fff3e3", SLATE: "#f3f4f6", GREEN: "#e8f6ec"}

W, H = 100, 56  # drawing units; 1 unit = 10 px at the default size
N_FRAMES, CYCLE, N_CALLS = 110, 24, 3


def clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def ramp(f: float, a: float, b: float) -> float:
    return clip01((f - a) / (b - a))


def extract_frames(video: Path, out: Path, size: int = 160, every: int = 20) -> list[np.ndarray]:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
                    f"select='not(mod(n\\,{every}))',scale={size}:{size}", "-vsync", "0", str(out / "%03d.png")],
                   check=True)
    return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(out.glob("*.png"))]


def fetch_episode_frames(work: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    from huggingface_hub import hf_hub_download

    paths = {n: Path(hf_hub_download(REPO_ID, f"{EPISODE}/{n}", repo_type="dataset", local_dir=work / "hub"))
             for n in ("rollout.mp4", "wrist.mp4")}
    return extract_frames(paths["rollout.mp4"], work / "agent"), extract_frames(paths["wrist.mp4"], work / "wrist")


# ------------------------------------------------------------------------------------------ drawing helpers
def text(ax, x, y, s, size=8.5, color=INK, ha="left", va="center", weight="normal", family=None, style="normal",
         alpha=1.0):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, family=family, style=style,
            alpha=alpha, clip_on=False)


def box(ax, x0, x1, y0, y1, *, active=False, color=BLUE, round_=1.2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={round_}",
                                fc=TINT[color] if active else BG, ec=color if active else LINE,
                                lw=1.8 if active else 1.1, zorder=1))


def arrow(ax, p0, p1, *, color=LINE, lw=1.4):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                                       mutation_scale=9), zorder=2)


def packet(ax, p0, p1, u, color, r=0.75):
    if 0.0 < u < 1.0:
        x, y = p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u
        ax.add_patch(plt.Circle((x, y), r, fc=color, ec=BG, lw=1.0, zorder=5))


def stage_title(ax, x, n, label):
    ax.add_patch(plt.Circle((x + 0.9, 48.6), 1.15, fc=INK, ec=INK, zorder=3))
    text(ax, x + 0.9, 48.6, str(n), 8, BG, "center", weight="bold")
    text(ax, x + 3.0, 48.6, label, 9.5, INK, weight="bold")


# ---------------------------------------------------------------------------------------------- one frame
OUT_ROWS = [  # (colour, key text, shape text, column, row)
    (BLUE, "base_image, wrist_image", "(P, 256, 2048)  blocks 16 + 8", 0, 0),
    (BLUE, "language", "(P, 200, 2048)  blocks 16 + 8", 0, 1),
    (BLUE, "pooled@all", "(P, 19, 6144)", 0, 2),
    (SLATE, "robot_state", "(P, 8)", 0, 3),
    (ORANGE, "expert@L0 ... L17", "(P, 3, 50, 1024)", 1, 0),
    (SLATE, "predicted_action_chunks", "(P, 50, 7)", 1, 1),
    (SLATE, "executed_actions + clocks", "(T, 7), (T,)", 1, 2),
    (SLATE, "rollout.mp4, wrist.mp4", "20 fps", 1, 3),
]
ROW_REVEAL = [26, 29, 32, 35, 38, 41, 44, 47]  # frame at which each output row is "written"


def draw_frame(f: int, agent: list[np.ndarray], wrist: list[np.ndarray], size_in=(9.0, 5.04), dpi=110):
    fig = plt.figure(figsize=size_in, dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")

    in_ship = f >= 72
    cyc = min(f // CYCLE, N_CALLS - 1) if f < 72 else N_CALLS - 1
    l = (f % CYCLE) if f < 72 else CYCLE  # position inside the per-call cycle (rest state after it)

    text(ax, 2, 53.6, "Collect, record and ship policy features", 13, INK, weight="bold")
    text(ax, 2, 51.0, "any VLA policy  |  LIBERO-X simulator  |  Hugging Face Hub", 8.5, GREY)

    A = (2, 22); B = (28, 54); C = (59, 76); D = (81, 98)
    y0, y1 = 26, 46
    a_act = f < 72 and (l < 7 or l >= 18)
    b_act = f < 72 and 6 <= l < 14
    c_act = f < 72 and (l >= 17 or (cyc > 0 and l < 3)) or (f >= 72 and f < 76)
    d_act = f >= 72

    # --- 1 simulator
    box(ax, *A, y0, y1, active=a_act, color=SLATE)
    stage_title(ax, A[0], 1, "Simulate")
    text(ax, A[0] + 1.2, 44.0, "task + init state", 7.8, GREY)
    text(ax, A[0] + 1.2, 41.9, '"turn on the flat stove"', 7.6, INK, style="italic")
    i = min(int(f / (N_FRAMES - 1) * (len(agent) - 1)), len(agent) - 1)
    ax.imshow(agent[i], extent=(A[0] + 1.0, A[0] + 10.0, 31.0, 40.0), zorder=3, interpolation="bilinear")
    ax.imshow(wrist[i], extent=(A[0] + 11.0, A[0] + 20.0, 31.0, 40.0), zorder=3, interpolation="bilinear")
    text(ax, A[0] + 5.5, 29.0, "agentview", 7.5, GREY, "center")
    text(ax, A[0] + 15.5, 29.0, "wrist", 7.5, GREY, "center")
    text(ax, A[0] + 1.2, 27.2, "proprio state", 7.5, GREY)

    # --- 2 policy
    box(ax, *B, y0, y1, active=b_act, color=BLUE)
    stage_title(ax, B[0], 2, "Policy")
    text(ax, B[0] + 9.6, 48.6, "any VLA, own venv", 7.8, GREY)
    pulse = b_act and 7 <= l < 13
    n_bars, bw, gap = 8, 2.3, 0.45
    xs0 = B[0] + 1.6
    hook_back, hook_exp = {2, 5}, {1, 4, 7}
    for r, (yb, label, col, hooks) in enumerate([(39.0, "backbone blocks", BLUE, hook_back),
                                                  (30.6, "action head blocks", ORANGE, hook_exp)]):
        for k in range(n_bars):
            x = xs0 + k * (bw + gap)
            hooked = k in hooks
            ax.add_patch(FancyBboxPatch((x, yb), bw, 4.0, boxstyle="round,pad=0,rounding_size=0.5",
                                        fc=col if hooked else "#e6e9ee", ec=col if hooked else "#c9ced6", lw=1.0,
                                        alpha=1.0 if (hooked and (pulse or not b_act)) else (0.95 if hooked else 1),
                                        zorder=3))
            if hooked:
                rr = 0.95 + (0.5 if pulse else 0.0)
                ax.add_patch(plt.Circle((x + bw / 2, yb + 4.0 + 1.3), rr * 0.7, fc=col, ec=BG, lw=1.0, zorder=4,
                                        alpha=1.0 if pulse else 0.85))
        text(ax, xs0, yb - 1.6, label + (" (frozen)" if r == 0 else ""), 7.6, GREY)
    text(ax, B[0] + 1.6, 27.2, "tap any block", 7.5, GREY)

    # --- 3 recorder
    box(ax, *C, y0, y1, active=c_act, color=GREEN)
    stage_title(ax, C[0], 3, "Record")
    text(ax, C[0] + 1.2, 44.0, f"policy call #{cyc + 1 if f < 72 else '...'}", 8, GREEN, weight="bold")
    for k, s in enumerate(["features", "robot state", "actions + clocks", "2 videos"]):
        text(ax, C[0] + 1.2, 41.2 - k * 2.5, "- " + s, 7.8, INK)
    ax.plot([C[0] + 1.0, C[1] - 1.0], [30.6, 30.6], color=LINE, lw=0.9, zorder=2)
    gate_ok = f >= 30
    text(ax, C[0] + 1.2, 28.9, ("alignment gate  ok" if gate_ok else "alignment gate"), 7.6,
         GREEN if gate_ok else GREY, weight="bold" if gate_ok else "normal")
    text(ax, C[0] + 1.2, 27.0, "-> manifest.csv", 7.6, GREY)

    # --- 4 hub
    box(ax, *D, y0, y1, active=d_act, color=BLUE)
    stage_title(ax, D[0], 4, "Ship")
    text(ax, D[0] + 1.2, 44.0, "Hugging Face Hub", 8, INK, weight="bold")
    text(ax, D[0] + 1.2, 41.7, "batch of ~50 GB", 7.6, GREY)
    px0, px1, py = D[0] + 1.2, D[1] - 1.2, 38.6
    ax.add_patch(Rectangle((px0, py), px1 - px0, 1.5, fc="#e6e9ee", ec="none", zorder=3))
    fill = ramp(f, 80, 92)
    ax.add_patch(Rectangle((px0, py), (px1 - px0) * fill, 1.5, fc=BLUE, ec="none", zorder=4))
    text(ax, px0, 36.3, "upload", 7.6, INK if fill > 0 else GREY)
    ver = f >= 94
    text(ax, px0, 33.9, ("verify: sha256  ok" if ver else "verify: sha256"), 7.6,
         GREEN if ver else GREY, weight="bold" if ver else "normal")
    dele = f >= 100
    text(ax, px0, 31.5, "delete local copy", 7.6, INK if dele else GREY, weight="bold" if dele else "normal")
    if dele:
        text(ax, px0, 29.1, "rollout.npz, *.mp4", 7.4, GREY)
        ax.plot([px0, px0 + 11.6], [29.1, 29.1], color=GREY, lw=1.0, zorder=5)
        text(ax, px0, 27.1, "kept: rollout.json", 7.4, GREY)

    # --- arrows + moving packets
    arrow(ax, (A[1] + 0.4, 38.0), (B[0] - 0.4, 38.0), color=SLATE)
    arrow(ax, (B[0] - 0.4, 32.0), (A[1] + 0.4, 32.0), color=SLATE)
    text(ax, (A[1] + B[0]) / 2, 39.5, "obs", 7.5, GREY, "center")
    text(ax, (A[1] + B[0]) / 2, 30.3, "actions", 7.5, GREY, "center")
    arrow(ax, (B[1] + 0.4, 40.0), (C[0] - 0.4, 40.0), color=BLUE)
    arrow(ax, (B[1] + 0.4, 34.0), (C[0] - 0.4, 34.0), color=ORANGE)
    arrow(ax, (C[1] + 0.4, 38.0), (D[0] - 0.4, 38.0), color=GREEN)
    if f < 72:
        packet(ax, (A[1] + 0.4, 38.0), (B[0] - 0.4, 38.0), ramp(l, 1, 6), SLATE)
        packet(ax, (B[0] - 0.4, 32.0), (A[1] + 0.4, 32.0), ramp(l, 14, 19), SLATE)
        packet(ax, (B[1] + 0.4, 40.0), (C[0] - 0.4, 40.0), ramp(l, 11, 16), BLUE)
        packet(ax, (B[1] + 0.4, 34.0), (C[0] - 0.4, 34.0), ramp(l, 11, 16), ORANGE)
    packet(ax, (C[1] + 0.4, 38.0), (D[0] - 0.4, 38.0), ramp(f, 72, 79), GREEN, r=0.9)

    # --- outputs panel
    ax.add_patch(FancyBboxPatch((2, 2.2), 96, 19.6, boxstyle="round,pad=0,rounding_size=1.2", fc="#fafbfc", ec=LINE,
                                lw=1.1, zorder=1))
    text(ax, 3.6, 19.6, "Outputs for pi0.5", 9.5, INK, weight="bold")
    text(ax, 3.6 + 19.5, 19.6, "rollout.npz, one row per policy call (P calls, T control steps)", 8, GREY)
    arrow(ax, ((C[0] + C[1]) / 2, y0 - 0.2), ((C[0] + C[1]) / 2, 22.2), color=GREEN, lw=1.2)
    for (col, key, shape, c, r), t_on in zip(OUT_ROWS, ROW_REVEAL):
        on = f >= t_on if f < 72 else True
        just = f < 72 and 0 <= f - t_on < 3
        x = 3.6 + c * 48.0
        y = 15.8 - r * 3.1
        ax.add_patch(FancyBboxPatch((x, y - 1.05), 2.1, 2.1, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=col if on else "#e6e9ee", ec="none", zorder=3))
        text(ax, x + 3.2, y, key, 8.2, INK if on else "#9aa1aa", family="DejaVu Sans Mono", weight="bold" if just else "normal")
        text(ax, x + 44.0, y, shape, 7.8, GREY if on else "#b7bcc4", "right")
    # legend of the tap colours
    for k, (col, lab) in enumerate([(BLUE, "backbone taps"), (ORANGE, "action-head taps"), (SLATE, "state, actions, video")]):
        xx = 3.6 + k * 20.5
        ax.add_patch(FancyBboxPatch((xx, 2.8), 1.3, 1.3, boxstyle="round,pad=0,rounding_size=0.3", fc=col, ec="none",
                                    zorder=3))
        text(ax, xx + 2.0, 3.45, lab, 7.4, GREY)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("pipeline.gif"))
    ap.add_argument("--still", type=Path, default=None, help="also write the final frame as a PNG")
    ap.add_argument("--only-frame", type=int, default=None, help="render one frame to --still and stop (layout checks)")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        agent, wrist = fetch_episode_frames(work)
        if args.only_frame is not None:
            Image.fromarray(draw_frame(args.only_frame, agent, wrist)).save(args.still or "frame.png")
            return 0
        frames = work / "frames"
        frames.mkdir()
        for f in range(N_FRAMES):
            Image.fromarray(draw_frame(f, agent, wrist)).save(frames / f"f{f:03d}.png")
        last = N_FRAMES - 1
        for k in range(int(args.fps * 2.0)):  # hold the final frame for 2 s
            (frames / f"f{N_FRAMES + k:03d}.png").write_bytes((frames / f"f{last:03d}.png").read_bytes())
        if args.still:
            (frames / f"f{last:03d}.png").replace(args.still)
            (frames / f"f{last:03d}.png").write_bytes(args.still.read_bytes())
        pal = work / "palette.png"
        base = ["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps), "-i", str(frames / "f%03d.png")]
        subprocess.run(base + ["-vf", "palettegen=max_colors=96:stats_mode=full", str(pal)], check=True)
        subprocess.run(base + ["-i", str(pal), "-lavfi", "paletteuse=dither=none", "-loop", "0", str(args.out)],
                       check=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
