# 一阶倒立摆 LQR 与 LoRA-LLM 控制实验

本项目使用 MATLAB R2023a 复现一阶倒立摆建模、LQR 控制、开环/闭环仿真、动画生成，以及 LoRA 微调 Llama 3.2 1B 直接控制实验；不涉及 Simulink 建模。

## 运行环境

- MATLAB R2023a
- Control System Toolbox：用于 `lqr`
- Symbolic Math Toolbox：用于 `derive_model.m` 的符号验证；缺少时脚本会记录跳过，但仍保存数值 A/B
- Python 3.9+
- Python 基础依赖：`numpy`, `matplotlib`, `scipy`, `pillow`
- Stage 3 LLM/LoRA 依赖：`unsloth`, `torch`, `bitsandbytes`, `transformers`, `trl`, `datasets`

Python 依赖安装：

```powershell
python -m pip install numpy matplotlib scipy pillow
python -m pip install unsloth torch bitsandbytes transformers trl datasets
```

## 安装指南
其实以下的东西你都不用看，你只需要复制这个仓库的地址，然后对任意有编码能力的ai说："我想本地部署这个仓库"。就这么简单！

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

也可以运行 MATLAB 侧批处理：

```matlab
run('src/run_all.m')
```

注意：`run_all.m` 只运行建模、阶段 1 和阶段 2，不会自动调用 Python 阶段 3。

## 主要输出

- `data/model_matrices.mat`：线性化 A/B、开环极点、符号验证结果
- `data/stage*_*.mat`：阶段 1-3 实验数据
- `data/*_log.txt`：阶段日志
- `data/stage3_lora_dataset.jsonl`：Stage 3 LoRA 训练数据，默认不纳入 Git
- `models/stage3_lora_controller/`：Stage 3 LoRA 控制器，默认不纳入 Git
- `models/stage3_lora_training_outputs/`：LoRA 训练中间输出，默认不纳入 Git
- `figures/*.png`：响应曲线
- `animations/*.gif`：二维动画

## 当前进度

- 阶段 1/2 已完成 MATLAB 建模、开环验证和 LQR 闭环基准。
- 阶段 3 已完成从“底座模型直接控制”到“本项目 LoRA 学习 LQR 控制律”的改造。
- LoRA 训练和 MATLAB 外置运行入口已经跑通；当前模型格式稳定，但数值回归误差仍会导致闭环失败。
- 下一步若继续优化 Stage 3，优先方向是扩充/重采样训练数据、降低 `llm_sample_time` 或改进连续动作表达方式。
