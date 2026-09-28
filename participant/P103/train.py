"""参赛模板训练脚本（学习基线示例）：SB3 PPO，训练环境与评测同源。

在训练专用 venv 内运行（torch/sb3/supersuit 仅训练 venv 安装，评测镜像不含）：
    .venv-train/Scripts/python.exe participant/P103/train.py --total-steps 1000000 \
        --curve-points 10 --checkpoint-steps 313344 626688 1044480 \
        --out outputs/P103/train-new
关键约束：训练栈不用 supersuit 的 flatten_observation_v0（其对 Discrete 字段做
one-hot 展开，与评测侧特征顺序不一致）；本文件 FlattenCoverageVecEnv 直接调用
coverage_bench.spaces.flatten_observation，训练与评测共用同一 flatten 实现。

实现说明（针对实际安装版本 stable-baselines3 2.9.0 / supersuit 3.11.0 的适配，
flatten 语义与 VecNormalize 设置与设计保持一致）：
1. AliveAgentsBridge：CoverageParallelEnv.possible_agents 按容量给出（agent_capacity），
   而每回合仅 config.num_agents 个 agent 在场；supersuit 的 pettingzoo_env_to_vec_env_v1
   要求 reset 后所有 possible_agents 均有观测，故在桥接处收拢 possible_agents。
2. FlattenCoverageVecEnv.seed/reset：supersuit 3.11 的 ConcatVecEnv 没有 seed() 方法，
   且 SB3 2.9 的 learn() 不再调用 env.seed()，种子只能经 reset(seed=...) 传入；
   因此 seed() 暂存种子，下一次 reset() 时经 ConcatVecEnv.reset(seed) 下发
   （ConcatVecEnv 会给第 i 个副本 seed+i，直达 CoverageParallelEnv.reset(seed=...)）。
3. RecordingVecMonitor：SB3 2.9 的 VecMonitor 不保留历史 episode 回报（也无
   get_episode_rewards()），本子类在 done 时从 info["episode"]["r"] 记录完成回报，
   供训练曲线取最近 100 个 episode 的均值。
4. 自动重置：每个环境副本从首次显式种子派生后续重置种子，保证整段训练可复现；
   保留终局 info，并把终局观测展平及标记时间截断，供 PPO 正确计算回合末价值。
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import supersuit as ss
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from coverage_bench.envs.factory import make_training_env
from coverage_bench.protocol import get_protocol_spec
from coverage_bench.spaces import flatten_observation, flattened_observation_size
from coverage_bench.suites import load_suite

_PUBLIC_SUITE = _REPO_ROOT / "configs" / "public-suite-v1.yaml"
TRAINING_SEED_DEFAULT = 7101  # 训练种子序列，与公开评测种子（public-seeds-v1.json）无关
_OUTPUTS_DIR = _REPO_ROOT / "outputs" / "P103"


class AliveAgentsBridge(BaseParallelWrapper):
    """把 possible_agents 对齐到场景实际存活的 agent（num_agents 个）。

    CoverageParallelEnv.possible_agents 按容量 agent_capacity 给出，而每回合只有
    config.num_agents 个 agent 在场；supersuit 的 pettingzoo_env_to_vec_env_v1
    要求 reset 后所有 possible_agents 都有观测，故在训练栈内做本桥接包装。
    """

    def __init__(self, env):
        super().__init__(env)
        self.possible_agents = [f"agent_{i}" for i in range(env.config.num_agents)]
        self._reset_rng = None

    def reset(self, seed=None, options=None):
        automatic_reset = seed is None and self._reset_rng is not None
        if seed is not None:
            self._reset_rng = np.random.default_rng(seed)
        elif automatic_reset:
            seed = int(self._reset_rng.integers(0, 2**63, dtype=np.int64))
        obs, infos = self.env.reset(seed=seed, options=options)
        self.possible_agents = list(self.env.agents)
        if automatic_reset:
            # SuperSuit 合并终局和重置 info；返回空字典，避免重置数据覆盖终局数据。
            infos = {agent: {} for agent in self.possible_agents}
        return obs, infos


class FlattenCoverageVecEnv(VecEnvWrapper):
    """把 agent 拆分后的 Dict 观测压平为与评测侧完全一致的 104 维向量。"""

    def __init__(self, venv, horizon: int):
        super().__init__(venv)
        self._spec = get_protocol_spec()
        self._horizon = int(horizon)
        size = flattened_observation_size(self._spec)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._pending_seed = None

    @staticmethod
    def _split_batch(obs):
        """把 gymnasium concatenate 的 dict-of-(n,...) 批量观测拆回 n 个单环境 Dict。"""
        num_envs = len(next(iter(obs.values())))
        return [{key: value[i] for key, value in obs.items()} for i in range(num_envs)]

    def seed(self, seed=None):
        """暂存种子，供下一次 reset() 消费（见模块 docstring 适配 2）。"""
        self._pending_seed = seed
        return [seed]

    def reset(self):
        if self._pending_seed is not None:
            # 绕过 SB3VecEnvWrapper（其 reset(seed) 会调用不存在的 ConcatVecEnv.seed），
            # 直接经 ConcatVecEnv.reset(seed=...) 把种子传给各副本的 pettingzoo 环境。
            obs = self.venv.venv.reset(seed=self._pending_seed)[0]
            self._pending_seed = None
        else:
            obs = self.venv.reset()
        # SB3 VecEnv 约定 reset 返回 (num_envs, obs_dim) 的 ndarray（VecNormalize 有断言）
        return np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        stacked = np.stack([flatten_observation(item, self._spec) for item in self._split_batch(obs)])
        for done, info in zip(dones, infos):
            if done:
                terminal = info.get("terminal_observation")
                if terminal is None:
                    raise RuntimeError("终局缺少 terminal_observation，无法正确处理时间截断")
                info["TimeLimit.truncated"] = bool(int(terminal["step_index"]) >= self._horizon)
                info["terminal_observation"] = flatten_observation(terminal, self._spec)
        return stacked, rewards, dones, infos


class RecordingVecMonitor(VecMonitor):
    """补记完成 episode 回报历史的 VecMonitor（SB3 2.9 的 VecMonitor 无此功能）。"""

    def __init__(self, venv):
        super().__init__(venv)
        self._completed_rewards = []

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        for i, done in enumerate(dones):
            if done and isinstance(infos[i], dict) and "episode" in infos[i]:
                self._completed_rewards.append(float(infos[i]["episode"]["r"]))
        return obs, rewards, dones, infos

    def get_episode_rewards(self):
        return self._completed_rewards


def build_env_stack(num_vec_envs: int, seed: int, case_id: str = "basic-0"):
    """构造到 FlattenCoverageVecEnv 为止的训练栈（不含 VecMonitor/VecNormalize）。"""
    if case_id == "mixed-basic0-coop0":
        from mixed_env import build_mixed_env_stack
        return build_mixed_env_stack(build_env_stack, num_vec_envs, seed)
    suite = load_suite(_PUBLIC_SUITE)
    matches = [case for group in suite.groups for case in group.cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ValueError(f"训练场景不存在或不唯一：{case_id}")
    case = matches[0]
    env = make_training_env(case.task_config)
    env = AliveAgentsBridge(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=num_vec_envs, num_cpus=0, base_class="stable_baselines3")
    env = FlattenCoverageVecEnv(env, horizon=case.task_config.horizon)
    env.seed(seed)
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(["git", *arguments], cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _checked_output_path(value: Path) -> Path:
    out_dir = (value if value.is_absolute() else _REPO_ROOT / value).resolve()
    root = _OUTPUTS_DIR.resolve()
    if out_dir == root or not out_dir.is_relative_to(root):
        raise ValueError(f"训练输出必须是 outputs/P103/ 下的独立子目录：{out_dir}")
    if out_dir.exists():
        raise FileExistsError(f"训练输出目录已存在，拒绝覆盖：{out_dir}")
    return out_dir


def _save_checkpoint(out_dir: Path, model: PPO, env: VecNormalize, meta: dict, requested_targets: list[int]) -> dict:
    actual_steps = int(model.num_timesteps)
    checkpoint_dir = out_dir / f"checkpoint-{actual_steps}"
    checkpoint_dir.mkdir(exist_ok=False)
    model.save(str(checkpoint_dir / "model"))
    # 网络与统计量在同一次 learn 返回后、下一次采样前保存，二者属于同一检查点。
    np.savez(
        checkpoint_dir / "vecnormalize-stats.npz",
        obs_mean=np.asarray(env.obs_rms.mean, dtype=np.float64),
        obs_var=np.asarray(env.obs_rms.var, dtype=np.float64),
        obs_eps=np.array(env.epsilon, dtype=np.float64),
        obs_clip=np.array(env.clip_obs, dtype=np.float64),
    )
    checkpoint_meta = {
        **meta,
        "total_steps": actual_steps,
        "wall_seconds": round(time.time() - meta["started_unix"], 1),
        "requested_checkpoint_targets": requested_targets,
    }
    checkpoint_meta.pop("started_unix")
    (checkpoint_dir / "training_meta.json").write_text(
        json.dumps(checkpoint_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        name: {"sha256": _sha256(checkpoint_dir / name), "size_bytes": (checkpoint_dir / name).stat().st_size}
        for name in ("model.zip", "vecnormalize-stats.npz", "training_meta.json")
    }
    record = {
        "actual_steps": actual_steps,
        "requested_targets": requested_targets,
        "directory": checkpoint_dir.relative_to(out_dir).as_posix(),
        "files": files,
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[checkpoint] actual_steps={actual_steps} targets={requested_targets} → {checkpoint_dir}")
    return record


def validate_training_args(args):
    if args.total_steps <= 0 or args.curve_points <= 0 or args.num_vec_envs <= 0 or args.n_steps <= 0:
        raise ValueError("训练步数、曲线段数、向量场景数和每轮步数都必须为正数")
    if args.batch_size <= 0 or args.seed < 0 or len(set(args.checkpoint_steps)) != len(args.checkpoint_steps):
        raise ValueError("batch_size/seed 无效，或检查点目标重复")
    rollout_samples = args.n_steps * args.num_vec_envs * 3
    steps_per_iter = max(args.total_steps // args.curve_points, 1)
    samples_per_iter = math.ceil(steps_per_iter / rollout_samples) * rollout_samples
    expected_actual_steps = math.ceil(args.total_steps / samples_per_iter) * samples_per_iter
    if any(step <= 0 or step > expected_actual_steps for step in args.checkpoint_steps):
        raise ValueError("检查点目标超出本次训练预算")
    protocol_path = None
    if args.protocol_path is not None:
        protocol_path = (args.protocol_path if args.protocol_path.is_absolute() else _REPO_ROOT / args.protocol_path).resolve()
        if not protocol_path.is_file() or not protocol_path.is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("固定协议文件必须位于 participant/P103 内且真实存在")
    return protocol_path, steps_per_iter, expected_actual_steps


def train(args):
    protocol_path, steps_per_iter, _expected_actual_steps = validate_training_args(args)
    out_dir = _checked_output_path(Path(args.out))
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    base_meta = {
        "seed": args.seed,
        "requested_total_steps": args.total_steps,
        "curve_points": args.curve_points,
        "started_unix": started,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "stable_baselines3_version": version("stable-baselines3"),
        "supersuit_version": version("supersuit"),
        "num_vec_envs": args.num_vec_envs,
        "suite": str(_PUBLIC_SUITE.relative_to(_REPO_ROOT)),
        "suite_sha256": _sha256(_PUBLIC_SUITE),
        "train_case": args.train_case,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": 0.95,
        "ent_coef": args.ent_coef,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "norm_obs": True,
        "norm_reward": False,
        "clip_obs": 10.0,
        "train_script_sha256": _sha256(Path(__file__)),
        "mixed_env_sha256": _sha256(Path(__file__).with_name("mixed_env.py")) if args.train_case == "mixed-basic0-coop0" else None,
        "source_commit": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--porcelain")),
        "protocol_path": protocol_path.relative_to(_REPO_ROOT).as_posix() if protocol_path else None,
        "protocol_sha256": _sha256(protocol_path) if protocol_path else None,
    }
    (out_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "seed": args.seed, "source_commit": base_meta["source_commit"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    flat_env = build_env_stack(args.num_vec_envs, args.seed, args.train_case)
    env = VecNormalize(RecordingVecMonitor(flat_env), norm_obs=True, norm_reward=False, clip_obs=10.0)
    try:
        model = PPO(
            "MlpPolicy", env,
            seed=args.seed,
            n_steps=args.n_steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
            gamma=args.gamma, gae_lambda=0.95, ent_coef=args.ent_coef, verbose=1,
        )
        pending = sorted(set(args.checkpoint_steps) | {args.total_steps})
        curve_rows = []
        checkpoints = []
        while model.num_timesteps < args.total_steps:
            # 保持 B1 冻结的分段推进方式；SB3 每次 learn 向上取整到整轮 rollout。
            model.learn(total_timesteps=steps_per_iter, reset_num_timesteps=False)
            ep_rewards = env.get_episode_rewards()
            mean100 = float(np.mean(ep_rewards[-100:])) if ep_rewards else float("nan")
            curve_rows.append((model.num_timesteps, mean100))
            print(f"[curve] steps={model.num_timesteps} mean_ep_reward(last100)={mean100:.4f}")
            reached = [target for target in pending if target <= model.num_timesteps]
            if reached:
                checkpoints.append(_save_checkpoint(out_dir, model, env, base_meta, reached))
                pending = [target for target in pending if target > model.num_timesteps]

        if pending:
            raise RuntimeError(f"训练结束时仍有未到达的检查点目标：{pending}")
        final = checkpoints[-1]
        final_dir = out_dir / final["directory"]
        for name in ("model.zip", "vecnormalize-stats.npz", "training_meta.json"):
            shutil.copyfile(final_dir / name, out_dir / name)
        with (out_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as output:
            output.write("total_steps,mean_ep_reward_last100\n")
            for steps, reward in curve_rows:
                output.write(f"{steps},{reward:.6f}\n")
        root_matches_final = all(
            _sha256(out_dir / name) == final["files"][name]["sha256"]
            for name in ("model.zip", "vecnormalize-stats.npz", "training_meta.json")
        )
        if not root_matches_final:
            raise RuntimeError("根目录模型和归一化统计量未与最终检查点保持一致")
        run_manifest = {
            "status": "complete",
            "requested_total_steps": args.total_steps,
            "actual_total_steps": model.num_timesteps,
            "checkpoints": checkpoints,
            "root_model_equals_final_checkpoint": root_matches_final,
        }
        (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "run_status.json").write_text(
            json.dumps({"status": "complete", "actual_total_steps": model.num_timesteps}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"训练完成: {model.num_timesteps} 步, {time.time() - started:.1f}s → {out_dir}")
    finally:
        env.close()


def check_export(args):
    """导出一致性检查：同一批环境观测上比较 SB3 原模型与 numpy 导出模型的确定性动作。

    覆盖观测预处理（flatten 同源函数）、归一化、网络前向、动作裁剪与 dtype 转换。
    """
    import torch

    model_dir = Path(args.model_dir)
    model = PPO.load(str(model_dir / "model.zip"), device="cpu")
    norm = np.load(model_dir / "vecnormalize-stats.npz")
    data = np.load(Path(args.npz))

    # 真实环境 rollout 收集未归一化的 104 维观测（与训练路径同栈、同 flatten）
    env = build_env_stack(args.num_vec_envs, args.seed, args.train_case)
    rng = np.random.default_rng(args.seed + 1)
    raw = []
    obs = env.reset()
    while len(raw) < args.num_obs:
        raw.extend(np.asarray(o, dtype=np.float64) for o in obs)
        actions = [rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) for _ in range(len(obs))]
        obs, _, _, _ = env.step(actions)
    raw = np.stack(raw[: args.num_obs])

    # SB3 路径：用训练侧统计量归一化（先除后 clip，与 SB3 VecNormalize.normalize_obs 一致）
    x_sb3 = np.clip(
        (raw - np.asarray(norm["obs_mean"], dtype=np.float64)) / np.sqrt(np.asarray(norm["obs_var"], dtype=np.float64) + float(norm["obs_eps"])),
        -float(norm["obs_clip"]), float(norm["obs_clip"]),
    )
    # numpy 路径：完全用 npz 内导出统计量，验证统计量逐位一致
    x_np = np.clip(
        (raw - data["obs_mean"]) / np.sqrt(data["obs_var"] + data["obs_eps"]),
        -data["obs_clip"], data["obs_clip"],
    )
    assert np.allclose(x_np, x_sb3, rtol=0.0, atol=1e-12), "npz 归一化统计量与训练侧统计量不一致"

    with torch.no_grad():
        sb3_actions, _ = model.predict(x_sb3, deterministic=True)
    sb3_actions = np.asarray(sb3_actions, dtype=np.float64)

    h = np.tanh(x_np @ data["W1"] + data["b1"])
    h = np.tanh(h @ data["W2"] + data["b2"])
    np_actions = np.clip(h @ data["W3"] + data["b3"], -1.0, 1.0)

    max_diff = float(np.max(np.abs(sb3_actions - np_actions)))
    print(f"样本数: {len(x_sb3)}  max|Δa|: {max_diff:.3e}  容差: 1e-5")
    if max_diff > 1e-5:
        raise SystemExit(f"导出一致性检查失败: max|Δa|={max_diff:.3e} > 1e-5")
    print("导出一致性检查通过")


def main():
    parser = argparse.ArgumentParser(description="参赛模板策略训练（SB3 PPO → npz）")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--num-vec-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED_DEFAULT)
    parser.add_argument("--train-case", choices=("basic-0", "basic-1", "coop-0", "coop-1", "mixed-basic0-coop0"), default="basic-0")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=[], help="累计样本检查点；须按实际训练预算预先指定")
    parser.add_argument("--out", type=Path, default=_OUTPUTS_DIR / "train-manual")
    parser.add_argument("--protocol-path", type=Path, default=None, help="本次实验的固定协议文件，仅记录其摘要")
    parser.add_argument("--check-export", action="store_true", help="运行导出一致性检查后退出")
    parser.add_argument("--model-dir", type=Path, default=_OUTPUTS_DIR / "train-manual")
    parser.add_argument("--npz", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "policy.npz")
    parser.add_argument("--num-obs", type=int, default=256)
    args = parser.parse_args()
    if args.check_export:
        check_export(args)
        return
    train(args)


if __name__ == "__main__":
    main()
