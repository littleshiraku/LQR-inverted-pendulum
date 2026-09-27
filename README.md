# LQR-inverted-pendulum

A cart-pole (first-order inverted pendulum) study in MATLAB: symbolic and numeric linearization, an LQR baseline, open-loop and closed-loop simulation with animation, and an optional Stage 3 experiment that fine-tunes Llama 3.2 1B with LoRA to imitate the LQR control law.

[English](README.md) | [简体中文](README.zh-CN.md)

## What Is Here

Two deliberately separated lines of work:

- **Stages 1-2 (MATLAB, LQR).** Derive the model, verify it, build the state-space matrices, design an LQR controller, and simulate open-loop and closed-loop responses with figures, animation and metrics.
- **Stage 3 (MATLAB + Python, LoRA-LLM).** Train a project-specific LoRA adapter on LQR expert data so the language model reproduces the controller's actions, then run the resulting controller closed-loop against the same simulation.

There is no Simulink modeling in this project.

## Environment

- MATLAB R2023a.
- Control System Toolbox - required for `lqr`.
- Symbolic Math Toolbox - used by `derive_model.m` for symbolic verification; if absent, the script records the skip and still saves the numeric A/B matrices.
- Python 3.9+.
- Base Python dependencies: `numpy`, `matplotlib`, `scipy`, `pillow`.

Stage 3 additionally needs an NVIDIA GPU environment with `unsloth`, `torch`, `bitsandbytes`, `transformers`, `trl` and `datasets`. Install the base set first:

```powershell
python -m pip install numpy matplotlib scipy pillow
```

Install the LoRA stack only if you intend to run Stage 3:

```powershell
python -m pip install unsloth torch bitsandbytes transformers trl datasets
```

## Running the Project

Set the MATLAB current folder to the project root. Run the stages in order, and prefer the base simulation before touching Stage 3.

### Stage 1-2: modeling and LQR baseline

```matlab
run('src/derive_model.m')
run('src/stage1_openloop.m')
run('src/stage2_lqr.m')
```

```matlab
run('src/run_all.m')
```

runs the same three scripts as a batch. **`run_all.m` does not run Stage 3** - it executes modeling, Stage 1 and Stage 2 only, then prints a reminder to invoke Stage 3 separately.

### Stage 3: LoRA controller (optional)

Train the adapter first. This shells out to an external Python process:

```matlab
run('src/train_stage3_lora_controller.m')
```

By default it generates 60,000 LQR expert samples for this project, clips the action range to +-60 N, and records MAE/RMSE of the LoRA output against the LQR action over validation states. Environment variables tune the scale and range:

```matlab
setenv('STAGE3_LORA_NUM_SAMPLES', '120000')
setenv('STAGE3_LORA_EPOCHS', '2')
setenv('STAGE3_LORA_U_MAX', '60')
setenv('STAGE3_LORA_VAL_SAMPLES', '256')
run('src/train_stage3_lora_controller.m')
```

Training artifacts go to `models/stage3_lora_controller/` and `data/stage3_lora_dataset.jsonl`; both are gitignored, so no weights or datasets are committed.

Then run Stage 3:

```matlab
run('src/run_stage3_llm_control.m')
```

It calls `python src/stage3_llm_control.py` by default. To point at a specific interpreter, set `STAGE3_PYTHON` first, for example `setenv('STAGE3_PYTHON', 'E:\conda\pyt\python.exe')`. Both the training and the run wrappers accept that variable; the Python script can also be executed directly.

**Execution status: not verified for Stage 3.** No MATLAB installation, GPU, or model weights were available when this README was written, so none of the commands above were run here. The parameter values and script names are read from the source, not from a successful run.

## Layout

```text
src/                        MATLAB and Python sources
scripts/                    build_classroom_ppt.js (classroom slide generation)
```

Generated at runtime and gitignored: `figures/`, `animations/`, `data/`, `models/`, `ppt.md`, `summary_report.md`, `LQR与LLM.pptx`.

The baseline LQR uses `Q = diag([15, 5, 180, 50])` and `R = 0.15`, defined in `src/common_params.m` together with the plant parameters (m = 1.0 kg, M = 5.0 kg, L = 2.0 m, g = 10.0 m/s^2), the 0.01 s sample time and the 30 s horizon.

## Outputs

- `data/model_matrices.mat` - linearized A/B, open-loop poles, symbolic verification result.
- `data/stage*_*.mat` - Stage 1-3 experiment data; `data/*_log.txt` - stage logs.
- `data/stage3_lora_dataset.jsonl` - Stage 3 training data (not in Git).
- `models/stage3_lora_controller/` - the Stage 3 LoRA controller (not in Git).
- `models/stage3_lora_training_outputs/` - LoRA training intermediates (not in Git).
- `figures/*.png` - response curves; `animations/*.gif` - 2D animation.

## Status and Limitations

- Stages 1 and 2 are complete: MATLAB modeling, open-loop verification and the LQR closed-loop baseline.
- Stage 3 has been reworked from "the base model controls the plant directly" to "a project-specific LoRA learns the LQR control law".
- LoRA training and the external MATLAB run entry point work, and the model format is stable. **Numeric regression error still causes the closed loop to fail**; the Stage 3 controller does not currently hold the pendulum.
- If Stage 3 is continued, the documented priorities are more or resampled training data, a smaller `llm_sample_time`, and a better continuous-action representation.

## License

Original contributions by littleshiraku in this repository, including original code and documentation, are licensed under the [MIT License](LICENSE). You may use, modify and redistribute these contributions, including commercially, provided that you retain the copyright and license notice. They are provided without warranty.

Third-party code, documents and other materials retain their respective copyrights and licenses. The root MIT license does not relicense third-party material or grant rights that littleshiraku does not hold.

MATLAB and its toolboxes, Python dependencies, and Llama model weights are governed by their own licenses; the MIT license here covers the original project contributions, not those products or model weights.
