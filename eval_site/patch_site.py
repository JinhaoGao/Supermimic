"""Add SuperMimic (policy_type supermimic) to the NOetix_GMT_eval site. Run from the repo root.
Exact-string edits with asserts; re-running is refused (idempotence guard)."""
import hashlib
import sys
from pathlib import Path

ROOT = Path.cwd()
EV = ROOT / "vendor/Noetix_GMT/scripts/evaluate_mujoco.py"
PLUGIN = ROOT / "vendor/Noetix_GMT/scripts/eval_supermimic.py"


STAGED = {}  # path -> new text; written only after every edit matched (all or nothing)


def edit(path, pairs):
    s = STAGED.get(path) or path.read_text(encoding="utf-8")
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit(f"{path}: expected exactly one match, found {n}:\n{old}")
        s = s.replace(old, new)
    STAGED[path] = s


if "supermimic" in EV.read_text(encoding="utf-8"):
    sys.exit("evaluate_mujoco.py already mentions supermimic; refusing to patch twice")
digest = hashlib.sha256(PLUGIN.read_bytes()).hexdigest()

# ---- evaluator ---------------------------------------------------------------------------
edit(EV, [
    ("""        _ANCHOR_ORI_B_MODULE = module
    return _ANCHOR_ORI_B_MODULE


""",
     f"""        _ANCHOR_ORI_B_MODULE = module
    return _ANCHOR_ORI_B_MODULE


# SuperMimic trackers (policy_type "supermimic", arXiv 2610.28480 reproduction): ONNX inputs
# obs_hist [1,H,3nj+6] + goal [1,K,2nj+6]. The input builder lives in eval_supermimic.py next to
# this file, loaded on first use by SuperMimicRunner and pinned by sha256 for the same reason as
# eval_anchor_ori_b.py: after editing it, update _SUPERMIMIC_SHA256 (this changes the cache key).
_SUPERMIMIC_SHA256 = "{digest}"
_SUPERMIMIC_MODULE = None


def _supermimic():
    global _SUPERMIMIC_MODULE
    if _SUPERMIMIC_MODULE is None:
        import hashlib
        import types

        path = Path(__file__).resolve().with_name("eval_supermimic.py")
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != _SUPERMIMIC_SHA256:
            raise ImportError(
                f"{{path}} sha256 {{digest}} != pinned _SUPERMIMIC_SHA256 {{_SUPERMIMIC_SHA256}}; "
                "update the pin in evaluate_mujoco.py (it keys the replay cache)"
            )
        module = types.ModuleType("_gmt_eval_supermimic")
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)  # exactly the bytes hashed above
        _SUPERMIMIC_MODULE = module
    return _SUPERMIMIC_MODULE


"""),
    ("""def create_runner_resources(cfg: RolloutCfg) -> RunnerResources:
    model = mujoco.MjModel.from_xml_path(cfg.model)
    model.opt.timestep = cfg.sim_dt
    resources = RunnerResources(model=model)
    if cfg.policy_type == "rgmt":
        if cfg.policy is None:
            raise ValueError("--policy is required for --policy-type rgmt")
""",
     """class SuperMimicRunner(BaseMujocoRunner):
    \"\"\"SuperMimic tracker: one ONNX with inputs obs_hist + goal; see eval_supermimic.py.\"\"\"

    def __init__(self, cfg: RolloutCfg, motion: MotionData, resources: RunnerResources | None = None) -> None:
        if cfg.policy is None:
            raise ValueError("--policy is required for --policy-type supermimic")
        self.policy_path = cfg.policy
        if resources is not None and resources.rgmt_session is not None:
            self.metadata = resources.rgmt_metadata or {}
            self.session = resources.rgmt_session
        else:
            self.metadata = load_onnx_metadata(self.policy_path)
            self.session = make_ort_session(self.policy_path, cfg.device)
        super().__init__(cfg, motion, resources)
        self._configure_robot_metadata(self.metadata)
        _supermimic().configure(self)

    def reset(self, start_frame: int) -> None:
        super().reset(start_frame)
        _supermimic().reset(self)

    def act(self, timestep: int) -> np.ndarray:
        return _supermimic().act(self, timestep)


def create_runner_resources(cfg: RolloutCfg) -> RunnerResources:
    model = mujoco.MjModel.from_xml_path(cfg.model)
    model.opt.timestep = cfg.sim_dt
    resources = RunnerResources(model=model)
    if cfg.policy_type in ("rgmt", "supermimic"):  # one ONNX; SuperMimic reuses the RGMT session slots
        if cfg.policy is None:
            raise ValueError(f"--policy is required for --policy-type {cfg.policy_type}")
"""),
    ("""    if cfg.policy_type == "external":
        return ExternalRunner(cfg, motion, resources)
    raise ValueError(f"Unsupported policy_type: {cfg.policy_type}")
""",
     """    if cfg.policy_type == "external":
        return ExternalRunner(cfg, motion, resources)
    if cfg.policy_type == "supermimic":
        return SuperMimicRunner(cfg, motion, resources)
    raise ValueError(f"Unsupported policy_type: {cfg.policy_type}")
"""),
    ("""def discover_tracked_body_names(cfg: RolloutCfg) -> list[str]:
    if cfg.policy_type == "rgmt" and cfg.policy:
""",
     """def discover_tracked_body_names(cfg: RolloutCfg) -> list[str]:
    if cfg.policy_type in ("rgmt", "supermimic") and cfg.policy:
"""),
    ("""choices=["rgmt", "sonic", "external"], default="rgmt", help="Policy export family.")""",
     """choices=["rgmt", "sonic", "external", "supermimic"], default="rgmt", help="Policy export family.")"""),
    ("""    if args.policy_type == "rgmt" and not args.policy:
        parser.error("--policy is required for --policy-type rgmt")
""",
     """    if args.policy_type in ("rgmt", "supermimic") and not args.policy:
        parser.error(f"--policy is required for --policy-type {args.policy_type}")
"""),
])

# ---- backend: profile schema ---------------------------------------------------------------
edit(ROOT / "backend/noetix_eval/config.py", [
    ("""    policy_type: Literal["rgmt", "sonic", "external"]
""",
     """    # "supermimic": SuperMimic tracker, one ONNX with inputs obs_hist [H, 3nj+6] and goal [K, 2nj+6]
    # (groups of the same names); see vendor/Noetix_GMT/scripts/eval_supermimic.py.
    policy_type: Literal["rgmt", "sonic", "external", "supermimic"]
"""),
    ("""        elif self.runner.policy_type == "external":
            # One group per ONNX input, named like the input; sizes are checked at registration.
            pass
""",
     """        elif self.runner.policy_type == "external":
            # One group per ONNX input, named like the input; sizes are checked at registration.
            pass
        elif self.runner.policy_type == "supermimic":
            if set(groups) != {"obs_hist", "goal"}:
                raise ValueError("SuperMimic profiles need exactly the groups obs_hist and goal")
            nj = self.expected_action_dim
            if groups["obs_hist"].frame_dim != 3 * nj + 6 or groups["goal"].frame_dim != 2 * nj + 6:
                raise ValueError(f"SuperMimic frames are obs_hist {3 * nj + 6} and goal {2 * nj + 6} wide")
            if groups["obs_hist"].frames != self.runner.history_steps:
                raise ValueError("SuperMimic obs_hist does not match runner.history_steps")
            if self.runner.rotation_6d_layout != "matrix_3x2_row_major":
                raise ValueError("SuperMimic goals use matrix_3x2_row_major rot6d")
"""),
])

# ---- backend: upload-time ONNX inspection --------------------------------------------------
edit(ROOT / "backend/noetix_eval/catalog.py", [
    ("""    if output_dim != observation.expected_action_dim:
        compatibility_errors.append(f"output dim {output_dim} != {observation.expected_action_dim}")
    if encoder is not None:
""",
     """    if observation.runner.policy_type == "supermimic":
        expected_shapes = {group.name: (group.frames, group.frame_dim) for group in observation.groups}
        if set(input_shapes) != set(expected_shapes):
            compatibility_errors.append(f"ONNX inputs {sorted(input_shapes)} != {sorted(expected_shapes)}")
        else:
            compatibility_errors.extend(
                f"input {name} shape {input_shapes[name]} != {shape}"
                for name, shape in expected_shapes.items()
                if input_shapes[name] != shape
            )
        if "anchor_body_name" not in metadata:
            compatibility_errors.append("missing metadata: anchor_body_name (SuperMimic goal orientation)")
    if output_dim != observation.expected_action_dim:
        compatibility_errors.append(f"output dim {output_dim} != {observation.expected_action_dim}")
    if encoder is not None:
"""),
])

# ---- backend: upload endpoint (one ONNX, stored like RGMT) ---------------------------------
edit(ROOT / "backend/noetix_eval/api.py", [
    ("""        bundle_hash = (
            component_hashes["decoder"]
            if observation.runner.policy_type == "rgmt"
            else stable_hash(component_hashes)
        )
""",
     """        bundle_hash = (
            component_hashes["decoder"]
            if observation.runner.policy_type in ("rgmt", "supermimic")
            else stable_hash(component_hashes)
        )
"""),
    ("""        primary_role = "policy" if observation.runner.policy_type == "rgmt" else "decoder"
""",
     """        primary_role = "policy" if observation.runner.policy_type in ("rgmt", "supermimic") else "decoder"
"""),
])

# ---- backend: web replay worker -------------------------------------------------------------
edit(ROOT / "backend/noetix_eval/worker/capture_rollout.py", [
    ("""    parser.add_argument("--policy-type", choices=["rgmt", "sonic"], required=True)
""",
     """    parser.add_argument("--policy-type", choices=["rgmt", "sonic", "supermimic"], required=True)
"""),
    ("""    policy_args = ["--policy", args.policy] if args.policy_type == "rgmt" else [
""",
     """    policy_args = ["--policy", args.policy] if args.policy_type in ("rgmt", "supermimic") else [
"""),
])

# ---- benchmarks: allow the two profiles -----------------------------------------------------
for bench, obs_id, family in [
    ("config/benchmarks/kimodo_g1_motion_official_v3.yaml", "supermimic_g1_v1", "supermimic_g1"),
    ("config/benchmarks/noetix_e2_motion_official_v1.yaml", "e2_supermimic_v1", "e2_supermimic"),
]:
    path = ROOT / bench
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        if line.startswith("observation_spec_ids: [") and line.rstrip().endswith("]"):
            line = line.rstrip()[:-1] + f", {obs_id}]\n"
        elif line.startswith("policy_families: [") and line.rstrip().endswith("]"):
            line = line.rstrip()[:-1] + f", {family}]\n"
        out.append(line)
    if sum(a != b for a, b in zip(out, lines)) != 2:
        sys.exit(f"{bench}: flow-style observation_spec_ids/policy_families lines not found")
    STAGED[path] = "".join(out)
for path, text in STAGED.items():
    path.write_text(text, encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}")
print(f"pinned eval_supermimic.py sha256 {digest}")
