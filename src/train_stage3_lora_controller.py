"""Train the Stage 3 LoRA controller from project-specific LQR data."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

from stage3_llm_control import (
    ALPACA_PROMPT,
    DATA_DIR,
    LLAMA_MAX_NEW_TOKENS,
    LLAMA_MAX_SEQ_LENGTH,
    LLAMA_MODEL_ID,
    LORA_MODEL_DIR,
    MODEL_CACHE_DIR,
    PARAMS,
    lqr_baseline_policy,
    rk4_step,
)


DATASET_PATH = Path(os.environ.get("STAGE3_LORA_DATASET_PATH", str(DATA_DIR / "stage3_lora_dataset.jsonl")))
TRAIN_LORA_MODEL_DIR = Path(os.environ.get("STAGE3_LORA_OUTPUT_DIR", str(LORA_MODEL_DIR)))
TRAINING_META_PATH = TRAIN_LORA_MODEL_DIR / "stage3_lora_training_meta.json"
BASE_MODEL_CACHE_DIR = (
    MODEL_CACHE_DIR
    / "hub"
    / "models--unsloth--Llama-3.2-1B-Instruct-unsloth-bnb-4bit"
    / "snapshots"
)

DEFAULT_NUM_SAMPLES = 60_000
DEFAULT_EPISODE_STEPS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 1.0
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_U_TRAIN_MAX = 60.0
DEFAULT_VAL_SAMPLES = 256

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def resolve_base_model_name() -> str:
    configured = os.environ.get("STAGE3_BASE_MODEL_PATH")
    if configured:
        return configured
    if BASE_MODEL_CACHE_DIR.exists():
        snapshots = sorted(path for path in BASE_MODEL_CACHE_DIR.iterdir() if (path / "config.json").exists())
        if snapshots:
            return str(snapshots[-1])
    return LLAMA_MODEL_ID


def format_state(state: np.ndarray) -> str:
    return (
        f"State: [x={state[0]:.3f}, dx={state[1]:.3f}, "
        f"theta={state[2]:.3f}, dtheta={state[3]:.3f}]"
    )


def generate_dataset(num_samples: int, episode_steps: int, u_train_max: float) -> dict:
    controller = lqr_baseline_policy()
    if controller is None:
        raise RuntimeError("SciPy is required to generate LQR expert data.")

    rng = np.random.default_rng(20260428)
    low = np.array([-0.5, -0.8, -0.18, -1.0], dtype=float)
    high = np.array([0.5, 0.8, 0.18, 1.0], dtype=float)
    hold_steps = max(1, int(round(PARAMS["llm_sample_time"] / PARAMS["dt"])))

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    episodes = 0
    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        while written < num_samples:
            state = rng.uniform(low=low, high=high)
            episodes += 1
            for _ in range(episode_steps):
                force = float(np.clip(controller(state.copy()), -u_train_max, u_train_max))
                record = {
                    "instruction": "You are a specialized LQR controller. Output only one line in this exact format: Action: <continuous force in Newtons>",
                    "input": format_state(state),
                    "output": f"Action: {force:.3f}",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                for _ in range(hold_steps):
                    state = rk4_step(state, force, PARAMS["dt"])
                if written >= num_samples:
                    break

    return {
        "dataset_path": str(DATASET_PATH),
        "num_samples": written,
        "episodes": episodes,
        "episode_steps": episode_steps,
        "state_low": low.tolist(),
        "state_high": high.tolist(),
        "expert": "continuous-time LQR on project dynamics",
        "u_train_max": u_train_max,
        "llm_sample_time": hold_steps * PARAMS["dt"],
        "llm_control_hold_steps": hold_steps,
    }


def parse_action(text: str) -> float | None:
    match = re.search(r"Action:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def validate_lora(model, tokenizer, val_samples: int, u_train_max: float) -> dict:
    if val_samples <= 0:
        return {"num_validation_samples": 0, "parse_failures": 0}

    controller = lqr_baseline_policy()
    if controller is None:
        raise RuntimeError("SciPy is required to validate LQR expert data.")

    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(20260429)
    low = np.array([-0.5, -0.8, -0.18, -1.0], dtype=float)
    high = np.array([0.5, 0.8, 0.18, 1.0], dtype=float)

    abs_errors = []
    sq_errors = []
    parse_failures = 0
    for _ in range(val_samples):
        state = rng.uniform(low=low, high=high)
        expert = float(np.clip(controller(state.copy()), -u_train_max, u_train_max))
        prompt = ALPACA_PROMPT.format(format_state(state))
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            use_cache=True,
            temperature=0.0,
            do_sample=False,
        )
        text_out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predicted = parse_action(text_out)
        if predicted is None:
            parse_failures += 1
            continue
        predicted = float(np.clip(predicted, -u_train_max, u_train_max))
        error = predicted - expert
        abs_errors.append(abs(error))
        sq_errors.append(error * error)

    valid = len(abs_errors)
    return {
        "num_validation_samples": val_samples,
        "valid_predictions": valid,
        "parse_failures": parse_failures,
        "mae": float(np.mean(abs_errors)) if valid else math.inf,
        "rmse": float(np.sqrt(np.mean(sq_errors))) if valid else math.inf,
        "max_abs_error": float(np.max(abs_errors)) if valid else math.inf,
    }


def train_lora(dataset_meta: dict) -> dict:
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    max_steps = env_int("STAGE3_LORA_MAX_STEPS", -1)
    batch_size = env_int("STAGE3_LORA_BATCH_SIZE", DEFAULT_BATCH_SIZE)
    grad_accum = env_int("STAGE3_LORA_GRAD_ACCUM", DEFAULT_GRAD_ACCUM)
    epochs = env_float("STAGE3_LORA_EPOCHS", DEFAULT_EPOCHS)
    learning_rate = env_float("STAGE3_LORA_LEARNING_RATE", DEFAULT_LEARNING_RATE)
    val_samples = env_int("STAGE3_LORA_VAL_SAMPLES", DEFAULT_VAL_SAMPLES)
    base_model_name = resolve_base_model_name()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=LLAMA_MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    eos_token = tokenizer.eos_token or ""

    def formatting_prompts_func(examples: dict) -> dict:
        texts = []
        for input_text, output_text in zip(examples["input"], examples["output"]):
            texts.append(ALPACA_PROMPT.format(input_text) + output_text + eos_token)
        return {"text": texts}

    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    output_dir = MODEL_CACHE_DIR / "stage3_lora_training_outputs"
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=LLAMA_MAX_SEQ_LENGTH,
        packing=False,
        report_to="none",
        save_strategy="no",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    validation_metrics = validate_lora(model, tokenizer, val_samples, dataset_meta["u_train_max"])

    TRAIN_LORA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(TRAIN_LORA_MODEL_DIR))
    tokenizer.save_pretrained(str(TRAIN_LORA_MODEL_DIR))

    training_meta = {
        **dataset_meta,
        "base_model_id": LLAMA_MODEL_ID,
        "base_model_name_or_path": base_model_name,
        "lora_model_dir": str(TRAIN_LORA_MODEL_DIR),
        "max_seq_length": LLAMA_MAX_SEQ_LENGTH,
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "validation_metrics": validation_metrics,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    TRAINING_META_PATH.write_text(json.dumps(training_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return training_meta


def main() -> None:
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))

    num_samples = env_int("STAGE3_LORA_NUM_SAMPLES", DEFAULT_NUM_SAMPLES)
    episode_steps = env_int("STAGE3_LORA_EPISODE_STEPS", DEFAULT_EPISODE_STEPS)
    u_train_max = env_float("STAGE3_LORA_U_MAX", DEFAULT_U_TRAIN_MAX)
    if not math.isfinite(u_train_max) or u_train_max <= 0 or u_train_max > PARAMS["u_max"]:
        raise ValueError(f"STAGE3_LORA_U_MAX must be in (0, {PARAMS['u_max']}], got {u_train_max!r}")

    print(f"Generating {num_samples} LQR expert samples with u_train_max={u_train_max:.3f} N...")
    dataset_meta = generate_dataset(num_samples, episode_steps, u_train_max)
    print(json.dumps(dataset_meta, ensure_ascii=False, indent=2))

    print("Training Stage 3 LoRA controller...")
    training_meta = train_lora(dataset_meta)
    print("Stage 3 LoRA training complete.")
    print(json.dumps(training_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
