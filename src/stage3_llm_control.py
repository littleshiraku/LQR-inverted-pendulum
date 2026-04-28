"""Stage 3: Llama 3.2 1B direct control.

The controller uses a local Unsloth 4-bit Llama model as the only source of
continuous force commands.  If the model output cannot be parsed as an action,
the simulation fails immediately instead of falling back to a heuristic policy.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
except Exception:  # pragma: no cover - script still writes numeric logs.
    plt = None
    PillowWriter = None

try:
    from scipy.io import savemat
    from scipy.linalg import solve_continuous_are
except Exception:  # pragma: no cover
    savemat = None
    solve_continuous_are = None


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
ANIM_DIR = ROOT / "animations"
DATA_DIR = ROOT / "data"
MODEL_CACHE_DIR = ROOT / "models"
for folder in (FIG_DIR, ANIM_DIR, DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))

LLAMA_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"
LLAMA_MAX_SEQ_LENGTH = 128
LLAMA_MAX_NEW_TOKENS = 16

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a specialized LQR controller. Output only one line in this exact format:
Action: <continuous force in Newtons>

### Input:
{}

### Response:
"""


PARAMS = {
    "m": 1.0,
    "M": 5.0,
    "L": 2.0,
    "g": 10.0,
    "dt": 0.01,
    "T": 30.0,
    "theta_fail": math.pi / 2,
    "theta_band": 0.01,
    "x_band": 0.02,
    "u_max": 200.0,
    "llm_sample_time": 0.1,
}


def integrate(y: np.ndarray, x: np.ndarray) -> float:
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(y, x))


def linear_matrices() -> tuple[np.ndarray, np.ndarray]:
    m, M, L, g = PARAMS["m"], PARAMS["M"], PARAMS["L"], PARAMS["g"]
    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, g * (M + m) / (M * L), 0.0],
        ]
    )
    b = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * L)]])
    return a, b


def dynamics(state: np.ndarray, force: float) -> np.ndarray:
    m, M, L, g = PARAMS["m"], PARAMS["M"], PARAMS["L"], PARAMS["g"]
    x_dot, theta, theta_dot = state[1], state[2], state[3]
    den = M + m - m * math.cos(theta) ** 2
    x_ddot = (
        force
        - m * g * math.sin(theta) * math.cos(theta)
        + m * L * math.sin(theta) * theta_dot**2
    ) / den
    theta_ddot = (g * math.sin(theta) - math.cos(theta) * x_ddot) / L
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)


def rk4_step(state: np.ndarray, force: float, dt: float) -> np.ndarray:
    k1 = dynamics(state, force)
    k2 = dynamics(state + 0.5 * dt * k1, force)
    k3 = dynamics(state + 0.5 * dt * k2, force)
    k4 = dynamics(state + dt * k3, force)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class LlamaDirectPolicy:
    """Unsloth Llama policy that maps state text directly to force text."""

    action_pattern = re.compile(r"Action:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

    def __init__(self) -> None:
        from unsloth import FastLanguageModel
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=LLAMA_MODEL_ID,
            max_seq_length=LLAMA_MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, state: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = state
        input_str = (
            f"State: [x={x:.3f}, dx={x_dot:.3f}, theta={theta:.3f}, "
            f"dtheta={theta_dot:.3f}]"
        )
        prompt = ALPACA_PROMPT.format(input_str)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.0,
            do_sample=False,
        )
        text_out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        match = self.action_pattern.search(text_out)
        if match is None:
            raise ValueError(f"LLM output did not contain a parseable Action line: {text_out!r}")
        force = float(match.group(1))
        if not math.isfinite(force):
            raise ValueError(f"LLM action is not finite: {force!r}")
        return force


def lqr_baseline_policy() -> Callable[[np.ndarray], float] | None:
    if solve_continuous_are is None:
        return None
    a, b = linear_matrices()
    q = np.diag([15.0, 5.0, 180.0, 50.0])
    r = np.array([[0.15]])
    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.solve(r, b.T @ p)

    def controller(state: np.ndarray) -> float:
        return float(np.clip(-(k @ state.reshape(-1, 1))[0, 0], -PARAMS["u_max"], PARAMS["u_max"]))

    controller.k = k  # type: ignore[attr-defined]
    return controller


def simulate(
    controller: Callable[[np.ndarray], float],
    x0: np.ndarray,
    label: str,
    control_hold_steps: int = 1,
) -> dict:
    if control_hold_steps < 1:
        raise ValueError("control_hold_steps must be at least 1.")
    dt, horizon = PARAMS["dt"], PARAMS["T"]
    t = np.arange(0.0, horizon + 0.5 * dt, dt)
    states = np.zeros((t.size, 4), dtype=float)
    forces = np.zeros(t.size, dtype=float)
    states[0] = x0
    failed = False
    failure_time = math.nan
    force = float("nan")
    for i in range(t.size - 1):
        if i % control_hold_steps == 0:
            force = controller(states[i].copy())
        forces[i] = force
        states[i + 1] = rk4_step(states[i], force, dt)
        if not np.all(np.isfinite(states[i + 1])) or abs(states[i + 1, 2]) > PARAMS["theta_fail"]:
            failed = True
            failure_time = float(t[i + 1])
            states[i + 2 :] = states[i + 1]
            forces[i + 1 :] = force
            break
    forces[-1] = force
    return {
        "label": label,
        "t": t,
        "x": states,
        "u": forces,
        "failed": failed,
        "failure_time": failure_time,
        "control_hold_steps": control_hold_steps,
    }


def settling_time(t: np.ndarray, y: np.ndarray, band: float) -> float:
    outside = np.flatnonzero(np.abs(y) > band)
    if outside.size == 0:
        return 0.0
    idx = int(outside[-1])
    if idx >= t.size - 1:
        return math.inf
    return float(t[idx + 1])


def metrics(result: dict) -> dict:
    t = result["t"]
    x = result["x"][:, 0]
    theta = result["x"][:, 2]
    u = result["u"]
    tail = slice(max(0, int(0.95 * t.size)), None)
    cost = integrate(10 * theta**2 + x**2 + 0.01 * u**2, t)
    if result["failed"]:
        cost += 1000.0
    return {
        "failed": bool(result["failed"]),
        "failure_time": float(result["failure_time"]) if result["failed"] else math.nan,
        "theta_settling_time": settling_time(t, theta, PARAMS["theta_band"]),
        "x_settling_time": settling_time(t, x, PARAMS["x_band"]),
        "theta_overshoot": float(max(0.0, np.max(np.abs(theta)) - abs(theta[0]))),
        "x_overshoot": float(max(0.0, np.max(np.abs(x)) - abs(x[0]))),
        "theta_steady_state_error": float(abs(np.mean(theta[tail]))),
        "x_steady_state_error": float(abs(np.mean(x[tail]))),
        "u_peak": float(np.max(np.abs(u))),
        "u_energy": integrate(u**2, t),
        "cost_j": cost,
        "reward": -cost,
    }


def plot_results(llm_result: dict, lqr_result: dict | None) -> None:
    if plt is None:
        return
    names = ["x (m)", "x_dot (m/s)", "theta (rad)", "theta_dot (rad/s)"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True)
    for i, name in enumerate(names):
        if lqr_result is not None:
            axes[i].plot(lqr_result["t"], lqr_result["x"][:, i], label="LQR baseline", color="#1f77b4")
        axes[i].plot(llm_result["t"], llm_result["x"][:, i], label="Llama 3.2 1B direct", color="#d62728")
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)
    if lqr_result is not None:
        axes[4].plot(lqr_result["t"], lqr_result["u"], label="LQR baseline", color="#1f77b4")
    axes[4].plot(llm_result["t"], llm_result["u"], label="Llama 3.2 1B direct", color="#d62728")
    axes[4].set_ylabel("u (N)")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle("Stage 3 Llama 3.2 1B direct control")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_llm_vs_lqr.png", dpi=160)
    plt.close(fig)


def animate(result: dict) -> None:
    if plt is None or PillowWriter is None:
        return
    t = result["t"]
    states = result["x"]
    frame_step = max(1, round(1 / (30 * PARAMS["dt"])))
    end = np.searchsorted(t, 8.0)
    indices = range(0, max(end, 1), frame_step)
    length = PARAMS["L"]
    x_min = float(np.min(states[:end, 0]) - length - 0.5)
    x_max = float(np.max(states[:end, 0]) + length + 0.5)

    fig, ax = plt.subplots(figsize=(9, 4.2))

    def update(idx: int) -> None:
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.4, length + 0.6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        x_cart, theta = states[idx, 0], states[idx, 2]
        cart_w, cart_h = 0.6, 0.28
        pivot = np.array([x_cart, cart_h / 2])
        bob = np.array([x_cart + length * math.sin(theta), cart_h / 2 + length * math.cos(theta)])
        ax.plot([x_min, x_max], [0, 0], "k-", linewidth=1)
        ax.add_patch(plt.Rectangle((x_cart - cart_w / 2, 0), cart_w, cart_h, color="#4c72b0", ec="k"))
        ax.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color="#dd8452", linewidth=4)
        ax.plot(pivot[0], pivot[1], "ko", markersize=5)
        ax.plot(bob[0], bob[1], "o", color="#c44e52", mec="k", markersize=14)
        ax.set_title(f"Llama 3.2 1B direct control, t={t[idx]:.2f}s")

    writer = PillowWriter(fps=30)
    with writer.saving(fig, ANIM_DIR / "stage3_llm_control.gif", dpi=100):
        for idx in indices:
            update(idx)
            writer.grab_frame()
    plt.close(fig)


def metric_table_md(metric: dict) -> str:
    def fmt(value: float) -> str:
        if isinstance(value, float) and math.isinf(value):
            return "Inf"
        if isinstance(value, float) and math.isnan(value):
            return "NaN"
        return f"{value:.6g}"

    rows = [
        ("theta 调节时间", f"{fmt(metric['theta_settling_time'])} s"),
        ("x 调节时间", f"{fmt(metric['x_settling_time'])} s"),
        ("theta 超调量", f"{fmt(metric['theta_overshoot'])} rad"),
        ("x 超调量", f"{fmt(metric['x_overshoot'])} m"),
        ("theta 稳态误差", f"{fmt(metric['theta_steady_state_error'])} rad"),
        ("x 稳态误差", f"{fmt(metric['x_steady_state_error'])} m"),
        ("控制量峰值", f"{fmt(metric['u_peak'])} N"),
        ("控制能量", fmt(metric["u_energy"])),
        ("代价 J", fmt(metric["cost_j"])),
        ("Reward", fmt(metric["reward"])),
        ("是否失败", str(metric["failed"])),
    ]
    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return "| 指标 | 数值 |\n|---|---:|\n" + body


def replace_stage_section(stage_title: str, section_body: str) -> None:
    ppt_path = ROOT / "ppt.md"
    if ppt_path.exists():
        text = ppt_path.read_text(encoding="utf-8")
    else:
        text = (
            "# 阶段 1：开环验证\n\n"
            "# 阶段 2：LQR 闭环\n\n"
            "# 阶段 3：LLM 直接控制\n\n"
            "# 阶段 4：LLM 调 Q 矩阵\n"
        )
    marker = f"# {stage_title}"
    next_mark = "\n# "
    start = text.find(marker)
    new_section = f"# {stage_title}\n\n{section_body.rstrip()}\n"
    if start < 0:
        text = text.rstrip() + "\n\n" + new_section
    else:
        end = text.find(next_mark, start + len(marker))
        if end < 0:
            text = text[:start] + new_section
        else:
            text = text[:start] + new_section + text[end + 1 :]
    ppt_path.write_text(text, encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(20260427)
    theta0 = float(rng.uniform(0.05, 0.15))
    x0 = np.array([0.0, 0.0, theta0, 0.0])
    llm_hold_steps = max(1, int(round(PARAMS["llm_sample_time"] / PARAMS["dt"])))
    llm_sample_time = llm_hold_steps * PARAMS["dt"]

    llm_policy = LlamaDirectPolicy()
    llm_result = simulate(llm_policy, x0, "Llama 3.2 1B direct", llm_hold_steps)
    llm_metrics = metrics(llm_result)

    lqr_controller = lqr_baseline_policy()
    lqr_result = simulate(lqr_controller, x0, "LQR baseline") if lqr_controller else None
    lqr_metrics = metrics(lqr_result) if lqr_result else None

    plot_results(llm_result, lqr_result)
    animate(llm_result)

    payload = {
        "theta0": theta0,
        "llm_metrics": llm_metrics,
        "lqr_metrics": lqr_metrics,
        "params": PARAMS,
        "llm_model_id": LLAMA_MODEL_ID,
        "llm_max_seq_length": LLAMA_MAX_SEQ_LENGTH,
        "llm_max_new_tokens": LLAMA_MAX_NEW_TOKENS,
        "llm_sample_time": llm_sample_time,
        "llm_control_hold_steps": llm_hold_steps,
    }
    (DATA_DIR / "stage3_llm_control_log.txt").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if savemat is not None:
        mat_payload = {
            "theta0": theta0,
            "t": llm_result["t"],
            "x_llm": llm_result["x"],
            "u_llm": llm_result["u"],
            "llm_failed": llm_result["failed"],
            "llm_sample_time": llm_sample_time,
            "llm_control_hold_steps": llm_hold_steps,
        }
        if lqr_result is not None:
            mat_payload.update({"x_lqr": lqr_result["x"], "u_lqr": lqr_result["u"]})
        savemat(DATA_DIR / "stage3_llm_control.mat", mat_payload)
    np.savez(DATA_DIR / "stage3_llm_control.npz", **llm_result)

    compare_text = ""
    if lqr_metrics is not None:
        delta = 100.0 * (lqr_metrics["cost_j"] - llm_metrics["cost_j"]) / abs(lqr_metrics["cost_j"])
        compare_text = f"\n- 相对 LQR 基准的代价变化：`{delta:.2f}%`（正值表示 Llama 直接控制代价更低）\n"

    body = (
        "## 实验目的\n"
        "将 Llama 3.2 1B 4-bit 指令模型作为直接策略网络，验证状态输入到连续控制力输出的闭环接口，并与阶段 2 LQR 基准对比。\n\n"
        "## 关键参数\n"
        f"- 蒙特卡罗次数：`1`\n- 随机初始角：`theta0={theta0:.5f} rad`\n"
        f"- 模型：`{LLAMA_MODEL_ID}`\n"
        f"- LLM 推理采样周期：`{llm_sample_time:.3f} s`，即每 `{llm_hold_steps}` 个 `dt` 更新一次控制力，中间零阶保持。\n"
        "- 策略形式：模型直接输出 `Action: <force>`，控制阶段不使用启发式、LQR 或零输出兜底。\n\n"
        "## 结果截图/动画\n"
        "![stage3](figures/stage3_llm_vs_lqr.png)\n\n"
        "动画：[`animations/stage3_llm_control.gif`](animations/stage3_llm_control.gif)\n\n"
        "## 性能指标\n"
        f"{metric_table_md(llm_metrics)}\n"
        f"{compare_text}\n"
        "## 结论与不足\n"
        "本阶段已接入真实 Llama 3.2 1B 4-bit 模型进行直接控制；未加载倒立摆 LoRA，因此控制性能完全取决于底座指令模型对数值动作格式的生成能力。\n"
    )
    replace_stage_section("阶段 3：LLM 直接控制", body)
    print("Stage 3 complete.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
