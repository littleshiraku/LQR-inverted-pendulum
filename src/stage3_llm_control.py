"""Stage 3: LoRA-tuned Llama 3.2 1B direct control.

The controller uses a project-specific LoRA adapter trained on LQR expert data.
If the model output cannot be parsed as an action, the simulation fails
immediately instead of falling back to a heuristic policy.
"""

from __future__ import annotations

# =============================================================================
# 1. 标准库与第三方依赖
# =============================================================================
# 本脚本既可以独立由 Python 运行，也可以被 MATLAB 包装脚本调用。
# 因此可选依赖采用 try/except 导入：缺少绘图或 MAT 文件写入能力时，
# 数值仿真和 JSON/NPZ 日志仍可继续执行。
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


# =============================================================================
# 2. 路径、模型与实验参数配置
# =============================================================================
# ROOT 指向项目根目录，后续所有输出目录都相对项目根目录生成。
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
ANIM_DIR = ROOT / "animations"
DATA_DIR = ROOT / "data"
MODEL_CACHE_DIR = ROOT / "models"

# 确保图像、动画、数据和模型缓存目录存在，避免后续写文件时报错。
for folder in (FIG_DIR, ANIM_DIR, DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 将 Hugging Face 缓存固定到项目 models 目录，便于复现实验和迁移工程。
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))

# 底座模型和 LoRA 控制器路径。
# STAGE3_LORA_PATH 可在外部覆盖，用于测试不同训练结果。
LLAMA_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"
LORA_MODEL_DIR = Path(os.environ.get("STAGE3_LORA_PATH", str(MODEL_CACHE_DIR / "stage3_lora_controller")))

# LLM 推理长度约束。动作只需要一行短文本，因此新生成 token 数保持较小。
LLAMA_MAX_SEQ_LENGTH = 128
LLAMA_MAX_NEW_TOKENS = 16

# Alpaca 风格提示词模板。控制阶段要求模型只输出一行 Action，方便正则解析。
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a specialized LQR controller. Output only one line in this exact format:
Action: <continuous force in Newtons>

### Input:
{}

### Response:
"""


# 倒立摆物理参数、仿真参数和评价阈值。
# 状态向量定义为 [x, x_dot, theta, theta_dot]。
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


# =============================================================================
# 3. 数值工具与倒立摆模型
# =============================================================================
def integrate(y: np.ndarray, x: np.ndarray) -> float:
    """使用梯形积分计算曲线面积，兼容不同 NumPy 版本的接口命名。"""
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(y, x))


def linear_matrices() -> tuple[np.ndarray, np.ndarray]:
    """返回直立平衡点附近的连续时间线性化矩阵 A、B。

    该线性模型用于构造 LQR 基准控制器，与非线性仿真中的 LLM 控制器对比。
    """
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
    """非线性倒立摆动力学。

    参数:
        state: 当前状态 [x, x_dot, theta, theta_dot]。
        force: 施加在小车上的水平控制力，单位 N。

    返回:
        状态导数 [x_dot, x_ddot, theta_dot, theta_ddot]。
    """
    m, M, L, g = PARAMS["m"], PARAMS["M"], PARAMS["L"], PARAMS["g"]
    x_dot, theta, theta_dot = state[1], state[2], state[3]

    # den 是非线性方程中小车加速度的等效质量项。
    den = M + m - m * math.cos(theta) ** 2

    # 先由小车方向动力学求 x_ddot，再代入摆杆角加速度方程。
    x_ddot = (
        force
        - m * g * math.sin(theta) * math.cos(theta)
        + m * L * math.sin(theta) * theta_dot**2
    ) / den
    theta_ddot = (g * math.sin(theta) - math.cos(theta) * x_ddot) / L
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)


def rk4_step(state: np.ndarray, force: float, dt: float) -> np.ndarray:
    """使用四阶 Runge-Kutta 方法推进一个仿真步长。"""
    k1 = dynamics(state, force)
    k2 = dynamics(state + 0.5 * dt * k1, force)
    k3 = dynamics(state + 0.5 * dt * k2, force)
    k4 = dynamics(state + dt * k3, force)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


# =============================================================================
# 4. LoRA Llama 直接控制策略
# =============================================================================
class LlamaDirectPolicy:
    """LoRA-tuned Unsloth Llama policy that maps state text to force text."""

    # 只接受形如 Action: -12.34 的输出，避免从其他解释文本中误读动作。
    action_pattern = re.compile(r"Action:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

    def __init__(self) -> None:
        """加载 LoRA 控制器、分词器和推理设备。"""
        if not LORA_MODEL_DIR.exists():
            raise FileNotFoundError(
                "Stage 3 LoRA controller was not found. Train it first with: "
                "run('src/train_stage3_lora_controller.m')"
            )

        # 控制力限幅优先读取训练元数据，防止推理时输出超过训练分布太多。
        self.force_limit = self._read_force_limit()

        from unsloth import FastLanguageModel
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        # 从 LoRA 目录加载合并后的推理模型，4-bit 加载降低显存占用。
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(LORA_MODEL_DIR),
            max_seq_length=LLAMA_MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, state: np.ndarray) -> float:
        """将当前连续状态格式化为文本，调用 LLM 输出控制力。"""
        x, x_dot, theta, theta_dot = state

        # 训练数据使用固定格式的状态文本；推理时保持一致可降低分布偏移。
        input_str = (
            f"State: [x={x:.3f}, dx={x_dot:.3f}, theta={theta:.3f}, "
            f"dtheta={theta_dot:.3f}]"
        )
        prompt = ALPACA_PROMPT.format(input_str)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # temperature=0 且 do_sample=False 表示确定性贪心解码，保证实验可复现。
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.0,
            do_sample=False,
        )
        text_out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # 若模型没有输出可解析动作，直接报错终止；本阶段不使用启发式兜底策略。
        match = self.action_pattern.search(text_out)
        if match is None:
            raise ValueError(f"LLM output did not contain a parseable Action line: {text_out!r}")
        force = float(match.group(1))
        if not math.isfinite(force):
            raise ValueError(f"LLM action is not finite: {force!r}")

        # 对控制力做饱和，保证输入不会超过训练/实验设定的物理约束。
        return float(np.clip(force, -self.force_limit, self.force_limit))

    def _read_force_limit(self) -> float:
        """从训练元数据读取控制力上限，缺失时退回全局 u_max。"""
        meta_path = LORA_MODEL_DIR / "stage3_lora_training_meta.json"
        if not meta_path.exists():
            return float(PARAMS["u_max"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        limit = float(meta.get("u_train_max", PARAMS["u_max"]))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"Invalid u_train_max in {meta_path}: {limit!r}")
        return min(limit, float(PARAMS["u_max"]))


# =============================================================================
# 5. LQR 基准控制器
# =============================================================================
def lqr_baseline_policy() -> Callable[[np.ndarray], float] | None:
    """构造 LQR 基准控制器；缺少 SciPy 时返回 None。"""
    if solve_continuous_are is None:
        return None
    a, b = linear_matrices()
    q = np.diag([15.0, 5.0, 180.0, 50.0])
    r = np.array([[0.15]])
    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.solve(r, b.T @ p)

    def controller(state: np.ndarray) -> float:
        # u = -Kx，并按统一最大控制力进行限幅。
        return float(np.clip(-(k @ state.reshape(-1, 1))[0, 0], -PARAMS["u_max"], PARAMS["u_max"]))

    # 将增益挂到函数对象上，便于调试或外部读取。
    controller.k = k  # type: ignore[attr-defined]
    return controller


# =============================================================================
# 6. 闭环仿真流程
# =============================================================================
def simulate(
    controller: Callable[[np.ndarray], float],
    x0: np.ndarray,
    label: str,
    control_hold_steps: int = 1,
) -> dict:
    """对给定控制器进行非线性闭环仿真。

    control_hold_steps 用于模拟 LLM 推理采样周期：每隔若干个积分步更新一次动作，
    中间步保持上一控制力，相当于零阶保持。
    """
    if control_hold_steps < 1:
        raise ValueError("control_hold_steps must be at least 1.")
    dt, horizon = PARAMS["dt"], PARAMS["T"]

    # 时间网格包含终点；states 和 forces 与 t 一一对应。
    t = np.arange(0.0, horizon + 0.5 * dt, dt)
    states = np.zeros((t.size, 4), dtype=float)
    forces = np.zeros(t.size, dtype=float)
    states[0] = x0
    failed = False
    failure_time = math.nan
    force = float("nan")
    for i in range(t.size - 1):
        # 到达控制采样点时重新调用控制器，否则沿用上一控制力。
        if i % control_hold_steps == 0:
            force = controller(states[i].copy())
        forces[i] = force
        states[i + 1] = rk4_step(states[i], force, dt)

        # 状态出现 NaN/Inf 或摆角超过失败阈值时，提前终止并保持最后状态。
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


# =============================================================================
# 7. 性能指标计算
# =============================================================================
def settling_time(t: np.ndarray, y: np.ndarray, band: float) -> float:
    """计算进入误差带后不再离开的调节时间。"""
    outside = np.flatnonzero(np.abs(y) > band)
    if outside.size == 0:
        return 0.0
    idx = int(outside[-1])
    if idx >= t.size - 1:
        return math.inf
    return float(t[idx + 1])


def metrics(result: dict) -> dict:
    """从仿真结果中提取统一性能指标。"""
    t = result["t"]
    x = result["x"][:, 0]
    theta = result["x"][:, 2]
    u = result["u"]

    # 用最后 5% 时间窗口估计稳态误差。
    tail = slice(max(0, int(0.95 * t.size)), None)

    # 简化二次代价：角度误差、小车位移和控制能量的加权积分。
    cost = integrate(10 * theta**2 + x**2 + 0.01 * u**2, t)
    if result["failed"]:
        # 失败轨迹额外加罚，便于与稳定轨迹在同一指标下比较。
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


# =============================================================================
# 8. 结果绘图与动画
# =============================================================================
def plot_results(llm_result: dict, lqr_result: dict | None) -> None:
    """绘制 LLM 直接控制与 LQR 基准的状态/控制量对比图。"""
    if plt is None:
        return
    names = ["x (m)", "x_dot (m/s)", "theta (rad)", "theta_dot (rad/s)"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 8), sharex=True)
    for i, name in enumerate(names):
        if lqr_result is not None:
            axes[i].plot(lqr_result["t"], lqr_result["x"][:, i], label="LQR baseline", color="#1f77b4")
        axes[i].plot(llm_result["t"], llm_result["x"][:, i], label="LoRA Llama direct", color="#d62728")
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)
    if lqr_result is not None:
        axes[4].plot(lqr_result["t"], lqr_result["u"], label="LQR baseline", color="#1f77b4")
    axes[4].plot(llm_result["t"], llm_result["u"], label="LoRA Llama direct", color="#d62728")
    axes[4].set_ylabel("u (N)")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle("Stage 3 LoRA Llama direct control")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_llm_vs_lqr.png", dpi=160)
    plt.close(fig)


def animate(result: dict) -> None:
    """将 LLM 闭环轨迹渲染为 GIF 动画。"""
    if plt is None or PillowWriter is None:
        return
    t = result["t"]
    states = result["x"]

    # 目标帧率 30 fps；根据积分步长抽帧，避免 GIF 文件过大。
    frame_step = max(1, round(1 / (30 * PARAMS["dt"])))

    # 动画只展示前 8 秒，和 MATLAB 侧动画保持一致。
    end = np.searchsorted(t, 8.0)
    indices = range(0, max(end, 1), frame_step)
    length = PARAMS["L"]
    x_min = float(np.min(states[:end, 0]) - length - 0.5)
    x_max = float(np.max(states[:end, 0]) + length + 0.5)

    fig, ax = plt.subplots(figsize=(9, 4.2))

    def update(idx: int) -> None:
        """绘制单帧小车-摆杆状态。"""
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.4, length + 0.6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        x_cart, theta = states[idx, 0], states[idx, 2]
        cart_w, cart_h = 0.6, 0.28

        # pivot 为摆杆铰接点，bob 为摆杆端点；theta=0 时摆杆竖直向上。
        pivot = np.array([x_cart, cart_h / 2])
        bob = np.array([x_cart + length * math.sin(theta), cart_h / 2 + length * math.cos(theta)])
        ax.plot([x_min, x_max], [0, 0], "k-", linewidth=1)
        ax.add_patch(plt.Rectangle((x_cart - cart_w / 2, 0), cart_w, cart_h, color="#4c72b0", ec="k"))
        ax.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color="#dd8452", linewidth=4)
        ax.plot(pivot[0], pivot[1], "ko", markersize=5)
        ax.plot(bob[0], bob[1], "o", color="#c44e52", mec="k", markersize=14)
        ax.set_title(f"LoRA Llama direct control, t={t[idx]:.2f}s")

    writer = PillowWriter(fps=30)
    with writer.saving(fig, ANIM_DIR / "stage3_llm_control.gif", dpi=100):
        for idx in indices:
            update(idx)
            writer.grab_frame()
    plt.close(fig)


# =============================================================================
# 9. 报告文本生成
# =============================================================================
def metric_table_md(metric: dict) -> str:
    """将指标字典转换为 Markdown 表格。"""
    def fmt(value: float) -> str:
        # Markdown 中显式保留 Inf/NaN，避免格式化成误导性的数字。
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
    """替换 ppt.md 中指定阶段章节；文件不存在时创建基础章节骨架。"""
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

    # 若目标章节不存在，则追加到文件末尾；否则只替换该章节到下一个一级标题之前的内容。
    if start < 0:
        text = text.rstrip() + "\n\n" + new_section
    else:
        end = text.find(next_mark, start + len(marker))
        if end < 0:
            text = text[:start] + new_section
        else:
            text = text[:start] + new_section + text[end + 1 :]
    ppt_path.write_text(text, encoding="utf-8")


# =============================================================================
# 10. 主流程入口
# =============================================================================
def main() -> None:
    """执行阶段 3 完整实验：初始化、仿真、对比、存档和报告写入。"""
    # 固定随机种子，使初始角度可复现。
    rng = np.random.default_rng(20260427)
    theta0 = float(rng.uniform(0.05, 0.15))
    x0 = np.array([0.0, 0.0, theta0, 0.0])

    # 将 LLM 采样周期换算成整数个积分步，实际采样时间以整数步为准。
    llm_hold_steps = max(1, int(round(PARAMS["llm_sample_time"] / PARAMS["dt"])))
    llm_sample_time = llm_hold_steps * PARAMS["dt"]

    # 加载 LoRA 控制器，并在非线性模型中仿真直接控制闭环。
    llm_policy = LlamaDirectPolicy()
    llm_result = simulate(llm_policy, x0, "LoRA Llama direct", llm_hold_steps)
    llm_metrics = metrics(llm_result)

    # 若 SciPy 可用，则同时生成 LQR 基准轨迹用于对比。
    lqr_controller = lqr_baseline_policy()
    lqr_result = simulate(lqr_controller, x0, "LQR baseline") if lqr_controller else None
    lqr_metrics = metrics(lqr_result) if lqr_result else None

    # 输出可视化文件：Python 侧 PNG 和 GIF。
    plot_results(llm_result, lqr_result)
    animate(llm_result)

    # JSON 日志保存实验配置、路径和指标，便于人工查看或脚本解析。
    payload = {
        "theta0": theta0,
        "llm_metrics": llm_metrics,
        "lqr_metrics": lqr_metrics,
        "params": PARAMS,
        "llm_model_id": LLAMA_MODEL_ID,
        "llm_lora_path": str(LORA_MODEL_DIR),
        "llm_force_limit": llm_policy.force_limit,
        "llm_max_seq_length": LLAMA_MAX_SEQ_LENGTH,
        "llm_max_new_tokens": LLAMA_MAX_NEW_TOKENS,
        "llm_sample_time": llm_sample_time,
        "llm_control_hold_steps": llm_hold_steps,
    }
    (DATA_DIR / "stage3_llm_control_log.txt").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # MAT 文件供 MATLAB 包装脚本读取并重新绘图/生成动画。
    if savemat is not None:
        mat_payload = {
            "theta0": theta0,
            "t": llm_result["t"],
            "x_llm": llm_result["x"],
            "u_llm": llm_result["u"],
            "llm_failed": llm_result["failed"],
            "llm_force_limit": llm_policy.force_limit,
            "llm_sample_time": llm_sample_time,
            "llm_control_hold_steps": llm_hold_steps,
        }
        if lqr_result is not None:
            mat_payload.update({"x_lqr": lqr_result["x"], "u_lqr": lqr_result["u"]})
        savemat(DATA_DIR / "stage3_llm_control.mat", mat_payload)

    # NPZ 文件保存 Python 原始结果，适合后续 NumPy 分析。
    np.savez(DATA_DIR / "stage3_llm_control.npz", **llm_result)

    compare_text = ""
    if lqr_metrics is not None:
        # 正值表示 LLM 代价低于 LQR；负值表示 LLM 代价更高。
        delta = 100.0 * (lqr_metrics["cost_j"] - llm_metrics["cost_j"]) / abs(lqr_metrics["cost_j"])
        compare_text = f"\n- 相对 LQR 基准的代价变化：`{delta:.2f}%`（正值表示 Llama 直接控制代价更低）\n"

    # 更新项目总报告中的阶段 3 章节。
    body = (
        "## 实验目的\n"
        "将 Llama 3.2 1B 4-bit 指令模型作为直接策略网络，验证状态输入到连续控制力输出的闭环接口，并与阶段 2 LQR 基准对比。\n\n"
        "## 关键参数\n"
        f"- 蒙特卡罗次数：`1`\n- 随机初始角：`theta0={theta0:.5f} rad`\n"
        f"- 底座模型：`{LLAMA_MODEL_ID}`\n"
        f"- LoRA 控制器：`{LORA_MODEL_DIR}`\n"
        f"- 控制力饱和：`±{llm_policy.force_limit:.3f} N`\n"
        f"- LLM 推理采样周期：`{llm_sample_time:.3f} s`，即每 `{llm_hold_steps}` 个 `dt` 更新一次控制力，中间零阶保持。\n"
        "- 策略形式：模型直接输出 `Action: <force>`，控制阶段不使用启发式、LQR 或零输出兜底。\n\n"
        "## 结果截图/动画\n"
        "![stage3](figures/stage3_llm_vs_lqr.png)\n\n"
        "动画：[`animations/stage3_llm_control.gif`](animations/stage3_llm_control.gif)\n\n"
        "## 性能指标\n"
        f"{metric_table_md(llm_metrics)}\n"
        f"{compare_text}\n"
        "## 结论与不足\n"
        "本阶段加载由本项目 LQR 专家数据训练得到的 LoRA 控制器；控制阶段仍只使用模型输出的连续力，不使用启发式或 LQR 兜底。\n"
    )
    replace_stage_section("阶段 3：LLM 直接控制", body)
    print("Stage 3 complete.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# 允许脚本被直接执行；被其他模块导入时不会自动运行实验。
if __name__ == "__main__":
    main()
