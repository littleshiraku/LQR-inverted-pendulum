# LQR-inverted-pendulum

一阶倒立摆的 MATLAB 研究项目：符号与数值线性化、LQR 基准控制器、开环与闭环仿真及动画，以及可选的阶段 3 实验——用 LoRA 微调 Llama 3.2 1B 来模仿 LQR 控制律。

[English](README.md) | [简体中文](README.zh-CN.md)

## 项目内容

两条刻意分离的工作线：

- **阶段 1—2（MATLAB，LQR）**：推导模型并验证，构建状态空间矩阵，设计 LQR 控制器，仿真开环与闭环响应，输出曲线、动画和指标。
- **阶段 3（MATLAB + Python，LoRA-LLM）**：在本项目生成的 LQR 专家数据上训练 LoRA 适配器，使语言模型复现控制器的动作，再用得到的控制器在同一套仿真中闭环运行。

本项目不涉及 Simulink 建模。

## 运行环境

- MATLAB R2023a。
- Control System Toolbox：`lqr` 所需。
- Symbolic Math Toolbox：`derive_model.m` 的符号验证使用；缺少时脚本会记录跳过，但仍保存数值 A/B 矩阵。
- Python 3.9+。
- Python 基础依赖：`numpy`、`matplotlib`、`scipy`、`pillow`。

阶段 3 额外需要可用的 NVIDIA GPU 环境与 `unsloth`、`torch`、`bitsandbytes`、`transformers`、`trl`、`datasets`。先安装基础依赖：

```powershell
python -m pip install numpy matplotlib scipy pillow
```

只有在需要运行阶段 3 时才安装 LoRA 依赖：

```powershell
python -m pip install unsloth torch bitsandbytes transformers trl datasets
```

## 运行方式

把 MATLAB 当前工作目录设置为项目根目录，按顺序运行各阶段。建议先跑通基础仿真，再考虑阶段 3。

### 阶段 1—2：建模与 LQR 基准

```matlab
run('src/derive_model.m')
run('src/stage1_openloop.m')
run('src/stage2_lqr.m')
```

```matlab
run('src/run_all.m')
```

会以批处理方式运行同样这三个脚本。**`run_all.m` 不运行阶段 3**，它只执行建模、阶段 1 和阶段 2，然后提示单独调用阶段 3。

### 阶段 3：LoRA 控制器（可选）

先训练适配器，这一步通过外置 Python 进程执行：

```matlab
run('src/train_stage3_lora_controller.m')
```

默认生成 60000 条本项目 LQR 专家样本，动作范围裁剪为 ±60 N，并在验证状态上记录 LoRA 输出相对 LQR 动作的 MAE/RMSE。可用环境变量调整规模和范围：

```matlab
setenv('STAGE3_LORA_NUM_SAMPLES', '120000')
setenv('STAGE3_LORA_EPOCHS', '2')
setenv('STAGE3_LORA_U_MAX', '60')
setenv('STAGE3_LORA_VAL_SAMPLES', '256')
run('src/train_stage3_lora_controller.m')
```

训练产物保存在 `models/stage3_lora_controller/`，训练数据保存在 `data/stage3_lora_dataset.jsonl`。两者都已被 `.gitignore` 忽略，不会提交模型权重或数据集。

然后执行阶段 3：

```matlab
run('src/run_stage3_llm_control.m')
```

默认调用 `python src/stage3_llm_control.py`。如需指定解释器，先设置 `STAGE3_PYTHON`，例如 `setenv('STAGE3_PYTHON', 'E:\conda\pyt\python.exe')`。训练和运行两个包装脚本都读取该变量，Python 脚本也可以直接运行。

**执行状态：阶段 3 未实测。** 撰写本 README 时没有可用的 MATLAB、GPU 或模型权重，因此上述命令均未在此处运行。参数取值和脚本名称来自源码，不是一次成功运行的记录。

## 目录结构

```text
src/                        MATLAB 与 Python 源码
scripts/                    build_classroom_ppt.js（课件生成）
```

运行时生成且被 Git 忽略：`figures/`、`animations/`、`data/`、`models/`、`ppt.md`、`summary_report.md`、`LQR与LLM.pptx`。

基准 LQR 使用 `Q = diag([15, 5, 180, 50])` 与 `R = 0.15`，与对象参数（m = 1.0 kg、M = 5.0 kg、L = 2.0 m、g = 10.0 m/s^2）、0.01 s 采样时间和 30 s 仿真时长一起定义在 `src/common_params.m` 中。

## 主要输出

- `data/model_matrices.mat`：线性化 A/B、开环极点、符号验证结果。
- `data/stage*_*.mat`：阶段 1—3 实验数据；`data/*_log.txt`：阶段日志。
- `data/stage3_lora_dataset.jsonl`：阶段 3 训练数据（不纳入 Git）。
- `models/stage3_lora_controller/`：阶段 3 LoRA 控制器（不纳入 Git）。
- `models/stage3_lora_training_outputs/`：LoRA 训练中间输出（不纳入 Git）。
- `figures/*.png`：响应曲线；`animations/*.gif`：二维动画。

## 状态与限制

- 阶段 1、2 已完成：MATLAB 建模、开环验证和 LQR 闭环基准。
- 阶段 3 已完成从"底座模型直接控制"到"本项目 LoRA 学习 LQR 控制律"的改造。
- LoRA 训练和 MATLAB 外置运行入口已经跑通，模型格式稳定。**数值回归误差仍会导致闭环失败**，阶段 3 控制器目前无法稳定摆杆。
- 若继续优化阶段 3，文档给出的优先方向是扩充/重采样训练数据、降低 `llm_sample_time`、改进连续动作表达方式。

## 许可

本仓库中 littleshiraku 的原创贡献（包括原创代码与文档）采用 [MIT 许可证](LICENSE)。在保留版权声明和许可证的前提下，可以使用、修改、再分发及用于商业用途；这些内容不提供任何担保。

第三方代码、文档和其他资料保留各自的版权及许可条款。根目录的 MIT 许可证不重新许可第三方内容，也不授予 littleshiraku 不拥有的权利。

MATLAB 及其工具箱、Python 依赖和 Llama 模型权重适用各自许可；此处 MIT 许可覆盖项目原创贡献，不覆盖这些产品或模型权重。
