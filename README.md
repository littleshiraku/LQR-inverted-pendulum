# 一阶倒立摆 LQR 与离线 LLM 控制实验

本项目使用MATLAB R2023a复现一阶倒立摆建模、LQR 控制、开环/闭环仿真、动画生成、离线 LLM 直接控制和离线 Q/R 自动调参实验，但不涉及任何simlink建模。

## 运行环境

- MATLAB R2023a
- Control System Toolbox：用于 `lqr`
- Symbolic Math Toolbox：用于 `derive_model.m` 的符号验证；缺少时脚本会记录跳过，但仍保存数值 A/B
- Optimization Toolbox / Statistics and Machine Learning Toolbox：可选，用于 `bayesopt`；缺少时 `stage4_tuneQ.m` 自动使用离线确定性搜索
- Python 3.9+
- Python 可选依赖：`numpy`, `matplotlib`, `scipy`, `pillow`

Python 依赖安装：

```powershell
python -m pip install numpy matplotlib scipy pillow
```

## 目录结构

```text
src/
  derive_model.m
  stage1_openloop.m
  stage2_lqr.m
  stage3_llm_control.py
  stage4_tuneQ.m
  run_all.m
figures/
animations/
data/
ppt.md
summary_report.md
```

`figures/`, `animations/`, `data/` 会由脚本自动创建。基准 LQR 使用 `Q=diag([15, 5, 180, 50])`, `R=0.15`

## 推荐运行顺序

在 MATLAB 当前工作目录设置为项目根目录后执行：

```matlab
run('src/derive_model.m')
run('src/stage1_openloop.m')
run('src/stage2_lqr.m')
```

然后在 PowerShell 执行阶段 3：

```powershell
python src/stage3_llm_control.py
```

最后回到 MATLAB 执行：

```matlab
run('src/stage4_tuneQ.m')
```

也可以运行 MATLAB 侧批处理：

```matlab
run('src/run_all.m')
```

注意：`run_all.m` 不会自动调用 Python 阶段 3。

## 离线 LLM 说明

阶段 3 和阶段 4 使用“离线模拟 LLM”：

- 阶段 3：目前使用确定性启发式策略模拟 qwen 策略网络接口，形式为 `state -> continuous force u`。
- 阶段 4：目前使用 `bayesopt` 或确定性候选搜索模拟 LLM 推荐 Q/R 的过程。

## 主要输出

- `data/model_matrices.mat`：线性化 A/B、开环极点、符号验证结果
- `data/stage*_*.mat`：各阶段实验数据
- `data/*_log.txt`：阶段日志
- `figures/*.png`：响应曲线和搜索曲线
- `animations/*.gif`：二维动画
