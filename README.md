# 一阶倒立摆 LQR 与 LoRA-LLM 控制实验

本项目使用MATLAB R2023a复现一阶倒立摆建模、LQR 控制、开环/闭环仿真、动画生成、LoRA 微调 Llama 3.2 1B 直接控制和离线 Q/R 自动调参实验，但不涉及任何 Simulink 建模。

## 运行环境

- MATLAB R2023a
- Control System Toolbox：用于 `lqr`
- Symbolic Math Toolbox：用于 `derive_model.m` 的符号验证；缺少时脚本会记录跳过，但仍保存数值 A/B
- Optimization Toolbox / Statistics and Machine Learning Toolbox：可选，用于 `bayesopt`；缺少时 `stage4_tuneQ.m` 自动使用离线确定性搜索
- Python 3.9+
- Python 基础依赖：`numpy`, `matplotlib`, `scipy`, `pillow`
- Stage 3 LLM/LoRA 依赖：`unsloth`, `torch`, `bitsandbytes`, `transformers`, `trl`, `datasets`

Python 依赖安装：

```powershell
python -m pip install numpy matplotlib scipy pillow
python -m pip install unsloth torch bitsandbytes transformers trl datasets
```

## 目录结构

```text
src/
  derive_model.m
  stage1_openloop.m
  stage2_lqr.m
  stage3_llm_control.py
  run_stage3_llm_control.m
  train_stage3_lora_controller.py
  train_stage3_lora_controller.m
  stage4_tuneQ.m
  run_all.m
figures/
animations/
data/
ppt.md
summary_report.md
```

`figures/`, `animations/`, `data/`, `models/` 会由脚本自动创建。基准 LQR 使用 `Q=diag([15, 5, 180, 50])`, `R=0.15`。

## 推荐运行顺序

在 MATLAB 当前工作目录设置为项目根目录后执行：

```matlab
run('src/derive_model.m')
run('src/stage1_openloop.m')
run('src/stage2_lqr.m')
```

首次运行阶段 3 前，先在 MATLAB 中通过外置 Python 进程训练本项目 LoRA 控制器：

```matlab
run('src/train_stage3_lora_controller.m')
```

默认训练会生成 `60000` 条本项目 LQR 专家样本，动作范围裁剪为 `±60 N`，训练后在验证状态上记录 LoRA 输出相对 LQR 动作的 MAE/RMSE。可用环境变量调整训练规模和动作范围，例如：

```matlab
setenv('STAGE3_LORA_NUM_SAMPLES', '120000')
setenv('STAGE3_LORA_EPOCHS', '2')
setenv('STAGE3_LORA_U_MAX', '60')
setenv('STAGE3_LORA_VAL_SAMPLES', '256')
run('src/train_stage3_lora_controller.m')
```

训练产物保存在 `models/stage3_lora_controller/`，训练数据保存在 `data/stage3_lora_dataset.jsonl`。这两个目录均被 `.gitignore` 忽略，不会提交模型权重或数据集。

然后执行阶段 3：

```matlab
run('src/run_stage3_llm_control.m')
```

默认调用 `python src/stage3_llm_control.py`。如果需要指定 Python 环境，可先设置环境变量 `STAGE3_PYTHON`，例如在 MATLAB 中执行 `setenv('STAGE3_PYTHON', 'E:\conda\pyt\python.exe')`。

最后回到 MATLAB 执行：

```matlab
run('src/stage4_tuneQ.m')
```

也可以运行 MATLAB 侧批处理：

```matlab
run('src/run_all.m')
```

注意：`run_all.m` 不会自动调用 Python 阶段 3。

## LLM 说明

阶段 3 使用真实本地 LLM 推理，阶段 4 仍使用离线推荐器：

- 阶段 3：先用本项目动力学和 LQR 专家控制律训练 `models/stage3_lora_controller/`，再加载该 LoRA 控制器输出 `Action: <force>`。动力学仍按 `dt=0.01 s` 积分，LLM 每 `0.1 s` 推理一次并对中间 10 个积分步保持上一控制力；控制力按训练元数据中的 `u_train_max` 做执行器饱和，控制阶段不使用启发式、LQR、零输出或解析失败兜底。
- 阶段 4：目前使用 `bayesopt` 或确定性候选搜索模拟 LLM 推荐 Q/R 的过程。

当前 Stage 3 的完整训练结果记录在 `models/stage3_lora_controller/stage3_lora_training_meta.json`：`60000` 条训练样本、`u_train_max=60 N`、验证集 `MAE≈8.13 N`、`RMSE≈13.74 N`、解析失败数为 `0`。该结果相比直接使用未微调底座模型更稳定，但仍不能达到 LQR 基准控制效果。

## 主要输出

- `data/model_matrices.mat`：线性化 A/B、开环极点、符号验证结果
- `data/stage*_*.mat`：各阶段实验数据
- `data/*_log.txt`：阶段日志
- `data/stage3_lora_dataset.jsonl`：Stage 3 LoRA 训练数据，默认不纳入 Git
- `models/stage3_lora_controller/`：Stage 3 LoRA 控制器，默认不纳入 Git
- `figures/*.png`：响应曲线和搜索曲线
- `animations/*.gif`：二维动画
