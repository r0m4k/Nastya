import os
import io
from pathlib import Path
import modal

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Meta-Llama-3.1-8B-Instruct "meta-llama/Meta-Llama-3.3-70B-Instruct"
VOLUME_NAME = "finetuning-volume"
CACHE_PATH = "/basemodel"
DATA_PATH = "/data"
OUTPUT_PATH = "/model_output"
new_adapter_name = "llama-3-8b-instruct-lora-adapter-jsonl" #  "llama-3-70b-instruct-lora-adapter-jsonl" "llama-3.1-8b-instruct-lora-adapter-jsonl" "llama-3.3-70b-instruct-lora-adapter-jsonl" 
# --- End of Configuration ---

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "datasets", "transformers", "peft", "trl",
    "bitsandbytes", "accelerate", "tensorboard"
)

app = modal.App("lora-finetune-app-jsonl", image=image)

volume = modal.NetworkFileSystem.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    gpu="H200",
    network_file_systems={
        CACHE_PATH: volume,
        DATA_PATH: volume,
        OUTPUT_PATH: volume,
    },
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=7200,
)
def train_lora_model():
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    )
    from trl import SFTTrainer

    # --- 1. Data Loading Section ---
    print("Searching for JSONL file(s) in volume...")
    data_files = [os.path.join(DATA_PATH, f.path) for f in volume.listdir(DATA_PATH) if f.path.endswith(".jsonl")]

    if not data_files:
        raise ValueError(f"Could not find any .jsonl files in {DATA_PATH} on the volume '{VOLUME_NAME}'.")

    print(f"Found data files: {data_files}")
    dataset = load_dataset("json", data_files=data_files, split="train")
    print("✅ Dataset loaded successfully from volume.")

    # --- 2. Tokenizer Loading (MOVED UP) ---
    # The tokenizer must be loaded before it can be used in the formatting function.
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, cache_dir=CACHE_PATH,
        trust_remote_code=True, token=os.environ["HF_TOKEN"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded successfully.")

    # --- 3. Llama 3 Chat Template Formatting Section ---
    # This function now has access to the 'tokenizer' defined above.
    def create_prompt(sample):
        prompt_template = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(prompt_template, tokenize=False)}

    # Apply the formatting function to the entire dataset
    dataset = dataset.map(create_prompt)
    print("✅ Dataset formatted with Llama 3 chat template.")

    # --- 4. Model Loading Section ---
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, cache_dir=CACHE_PATH,
        quantization_config=quant_config, device_map="auto",
        token=os.environ["HF_TOKEN"]
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print("✅ Model loaded successfully.")


    # --- 5. Training Configuration ---
    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64,
        bias="none", task_type="CAUSAL_LM"
    )

    output_dir = os.path.join(OUTPUT_PATH, "results")
    final_adapter_path = os.path.join(OUTPUT_PATH, new_adapter_name)

    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    # --- 6. SFTTrainer Initialization ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,         # Tells the trainer to set up LoRA adapters
        args=training_params,
        # You may also need to pass the tokenizer if the trainer doesn't pick it up automatically
        # tokenizer=tokenizer, 
    )

    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()
    print("✅ LoRA training complete.")

    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"✅ LoRA adapter saved to: {final_adapter_path} on the volume.")

@app.local_entrypoint()
def main():
    # Look for any .jsonl file in the current directory
    local_data_dir = Path(".")
    local_jsonl_files = list(local_data_dir.glob("*.jsonl"))
    if not local_jsonl_files:
        print("❌ No local '.jsonl' files found. Please place your dataset file (e.g., 'data.jsonl') in the same directory as the script.")
        return

    print(f" Detected {len(local_jsonl_files)} JSONL file(s). Uploading to volume...")

    # REMOVED THIS LINE: volume.mkdir(DATA_PATH, exist_ok=True)
    
    for local_path in local_jsonl_files:
        remote_path = Path(DATA_PATH) / local_path.name
        print(f" - Uploading '{local_path.name}' to '{remote_path}'")
        with open(local_path, "rb") as f:
            volume.write_file(str(remote_path), f) # This will create the directory if needed

    print("\n🚀 Launching remote LoRA fine-tuning job on Modal...")
    train_lora_model.remote()