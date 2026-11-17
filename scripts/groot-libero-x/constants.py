"""Shared constants + per-run config carrier for the GR00T-on-LIBERO-X pipeline.

LIBERO-X (``meituan/LIBERO-X``, pinned in ``submodules/LIBERO-X``) adds new
objects, textures, and goal predicates (``ExactIn``, ``UprightOn``, ``SideOn``)
across 5 difficulty levels. ``collect.py`` is the aligned rollout/feature
collector; it talks to the GR00T websocket server in ``server/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# scripts/groot-libero-x/ -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

LIBERO_X_ROOT = REPO_ROOT / "submodules" / "LIBERO-X"

LEVELS = ("LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5")

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "groot-libero-x"
DEFAULT_COLLECT_DIR = DEFAULT_OUTPUT_DIR / "collect"
# Default task lists: one BDDL filename per line (72 / 73 / 73 for LEVEL1-3).
TASKS_DIR = Path(__file__).resolve().parent / "tasks"

# The checkpoint fine-tuned on LIBERO-X (LEVEL1-3). collect.py targets this by default.
CUSTOM_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "gr00t-libero-x-level123"
CUSTOM_CHECKPOINT_LABEL = "rohansiva/gr00t-libero-x-level123"
MODEL_ID = "gr00t-n1.7-libero-x"

CONTROL_HZ = 20.0
DUMMY_ACTION = [0.0] * 6 + [-1.0]
ACTION_DIM = 7
# GR00T-N1.7 LIBERO_PANDA modality config emits a 16-step action horizon.
GROOT_ACTION_HORIZON = 16

SERVER_SCRIPT = "scripts/groot-libero-x/server/serve_groot_ws.py"
FEATURE_SERVER_SCRIPT = SERVER_SCRIPT

# Backbone feature capture (collect.py --with-features). Layer-16 hidden states
# of the Cosmos-Reason2-2B (Qwen3-VL) backbone. The fine-tune keeps LIBERO_PANDA,
# 256px images / 2x2 merge, and the frozen 28-layer backbone.
FEATURE_SELECT_LAYER = 16
GROOT_HIDDEN = 2048            # Qwen3-VL text hidden size
GROOT_IMG_TOKENS = 64         # 256px / patch 16 -> 16x16 -> 2x2 merge -> 8x8 per camera
LANG_MAX_TOKENS = 200         # instruction tokens zero-padded to this
STATE_FEATURE_DIM = 1536      # action head input_embedding_dim
FEATURE_SOURCE = "backbone"
FEATURE_MODULE = (
    "gr00t.model.modules.qwen3_backbone.Qwen3Backbone hidden_states[-1] "
    f"(select_layer={FEATURE_SELECT_LAYER}); state_features from Gr00tN1d7ActionHead"
)

# Aliases used by feature_schema.py.
NUM_IMAGE_TOKENS = GROOT_IMG_TOKENS
HIDDEN_DIM = GROOT_HIDDEN


@dataclass(frozen=True)
class FeatureSpec:
    """What a policy's feature server returns, so the collection pipeline can check/save it."""

    name: str
    model_id: str
    img_tokens: int          # tokens per camera
    hidden: int              # feature width
    lang_max: int            # language slots (zero-padded)
    state_dim: int | None    # None: the policy has no separate state feature
    source: str
    module: str
    server_script: str


GROOT_SPEC = FeatureSpec(
    name="groot", model_id=MODEL_ID, img_tokens=GROOT_IMG_TOKENS, hidden=GROOT_HIDDEN,
    lang_max=LANG_MAX_TOKENS, state_dim=STATE_FEATURE_DIM, source=FEATURE_SOURCE,
    module=FEATURE_MODULE, server_script=SERVER_SCRIPT,
)

# pi0.5 (LeRobot): block outputs of the PaliGemma Gemma-2B backbone over the prefix
# [image 256 | wrist 256 | empty-camera 256 | text 200]. img_tokens is 256 unless the server
# pools (--pool-image); collect.py rebuilds the spec from the server metadata.
PI05_SPEC = FeatureSpec(
    name="pi05", model_id="pi05-lerobot", img_tokens=256, hidden=2048, lang_max=200,
    state_dim=None, source="backbone",
    module="lerobot PI05 paligemma Gemma-2B block output on the prefix pass (layers: see server metadata)",
    server_script="scripts/pi05-libero-x/serve_pi05.py",
)


@dataclass
class EvalConfig:
    libero_x_root: Path = LIBERO_X_ROOT
    repo_root: Path = REPO_ROOT

    host: str = "127.0.0.1"
    port: int = 8000

    level: str = "LEVEL1"
    seed: int = 7
    n_tasks: int = 5
    n_rollouts: int = 10

    env_resolution: int = 256
    # LIBERO-X's own eval_template.py uses max_steps=1200, num_steps_wait=10,
    # replan_steps=25 -- its tasks are longer-horizon than libero_10 (many
    # need >500 steps), so matching those defaults is load-bearing, not
    # cosmetic.
    max_steps: int = 1200
    num_steps_wait: int = 10
    replan_steps: int = 25

    checkpoint: str = CUSTOM_CHECKPOINT_LABEL

    # --- collect.py (rollout/feature collection) per-episode carrier ---
    episode_index: int = 0
    task_name: str = ""      # BDDL stem
    task_index: int = 0      # 0-based position in the sampled/finetune task list
    save_video: bool = True
    checkpoint_label: str = ""  # overrides `checkpoint` in the rollout record when set

    @property
    def base_level(self) -> str:
        # LIBERO-X ships only 4 BDDL/init level dirs; LEVEL5 reuses LEVEL4's.
        return "LEVEL4" if self.level.upper() == "LEVEL5" else self.level.upper()

    @property
    def bddl_dir(self) -> Path:
        return self.libero_x_root / "libero" / "libero_x" / "bddl" / self.base_level

    @property
    def init_dir(self) -> Path:
        return self.libero_x_root / "libero" / "libero_x" / "init" / self.base_level

    @property
    def libero_config_dir(self) -> Path:
        return self.outputs_dir / ".libero"

    @property
    def outputs_dir(self) -> Path:
        return DEFAULT_OUTPUT_DIR
