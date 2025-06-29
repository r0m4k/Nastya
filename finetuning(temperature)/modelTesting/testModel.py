import os
import io
import csv
import time
from pathlib import Path
import modal

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Meta-Llama-3.1-8B-Instruct "meta-llama/Meta-Llama-3.3-70B-Instruct"
VOLUME_NAME = "finetuning-volume"
CACHE_PATH = "/basemodel"
DATA_PATH = "/data"
OUTPUT_PATH = "/model_output"

# IMPORTANT: This must match the name you used in the training script
ADAPTER_NAME = "llama-3-8b-lora-adapter-jsonl" # "llama-3-70b-lora-adapter-jsonl" "llama-3.1-8b-lora-adapter-jsonl" "llama-3.3-70b-lora-adapter-jsonl" "llama-3-8b-lora-adapter-jsonl" // "llama-3-8b-instruct-lora-adapter-jsonl" "llama-3-70b-instruct-lora-adapter-jsonl" "llama-3.1-8b-instruct-lora-adapter-jsonl" "llama-3.3-70b-instruct-lora-adapter-jsonl" 
ADAPTER_PATH = os.path.join(OUTPUT_PATH, ADAPTER_NAME)
# --- End of Configuration ---

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "transformers", "peft",
    "bitsandbytes", "accelerate", "scikit-learn"
)

app = modal.App("lora-inference-app", image=image)

volume = modal.NetworkFileSystem.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    gpu="H100",
    network_file_systems={
        CACHE_PATH: volume,
        DATA_PATH: volume,
        OUTPUT_PATH: volume,
    },
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=1800, # Increased timeout for testing multiple datasets
)
def test_model():
    """
    This function runs on a Modal GPU container. It loads the fine-tuned
    model and evaluates it against the validation datasets.
    """
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    )
    from sklearn.metrics import f1_score

    print(f"Loading base model: {BASE_MODEL_ID}")
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # 1. Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        cache_dir=CACHE_PATH,
        quantization_config=quant_config,
        device_map="auto",
        token=os.environ["HF_TOKEN"]
    )

    # 2. Load the LoRA adapter
    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # 3. Merge the adapter into the model for faster inference
    print("Merging adapter...")
    model = model.merge_and_unload()
    print("✅ Model and adapter merged successfully.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    datasets_to_test = ['laptop_14', 'restaurants_14', 'restaurants_15', 'restaurants_16']
    
    for dataset in datasets_to_test:
        validation_file_path = os.path.join(DATA_PATH, dataset, 'data_validation.csv')
        
        if not os.path.exists(validation_file_path):
            print(f"Warning: Could not find validation file: {validation_file_path}")
            continue

        with open(validation_file_path, 'r', newline='', encoding='utf-8') as infile:
            csv_reader = csv.DictReader(infile)
            
            sentiments_original = []
            sentiments_predicted = []

            print(f"\nProcessing dataset: {dataset}...")
            for row in csv_reader:
                sentence_text = row.get('sentence')
                term = row.get('aspect_term')
                original_polarity = row.get('sentiment')

                if not sentence_text or not term or not original_polarity:
                    continue
                
                # Using the same prompt structure you provided
                prompt = f"""
Instruction:
Analyze the sentiment of the aspect term within the given sentence. The aspect term is highlighted by quotes. Your answer must be one of the following three options: 'positive', 'negative', 'neutral'. Do not provide any explanation or other text.
Sentence:
"{sentence_text}"
Aspect Term:
"{term}"
Sentiment:
"""
                sentiments_original.append(original_polarity.strip().lower())

                # Prepare input for the model
                messages = [{"role": "user", "content": prompt}]
                tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

                # Generate the response
                outputs = model.generate(tokenized_chat, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id, temperature=0.0)
                
                # Decode only the newly generated tokens
                response_text = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True).strip().lower()
                sentiments_predicted.append(response_text)

        # --- Calculate Metrics ---
        correct_predictions = sum(
            1 for original, predicted in zip(sentiments_original, sentiments_predicted)
            if original == predicted
        )
        accuracy = correct_predictions / len(sentiments_original) * 100 if sentiments_original else 0

        # For F1, handle cases where the model predicts an unexpected value
        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        y_true = [sentiment_map[s] for s in sentiments_original]
        # Use .get() with a default value to avoid errors if the model outputs something unexpected
        y_pred = [sentiment_map.get(s, -2) for s in sentiments_predicted] 

        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"________________________")
        print(f"Model: Fine-tuned Llama-3-8B ({ADAPTER_NAME})")
        print(f"Dataset: {dataset}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"F1 Score: {f1:.4f}")

@app.local_entrypoint()
def main():
    """
    This function runs locally. It finds your validation data in sibling
    directories, uploads them, and then triggers the remote testing function.
    """
    # 1. Define the names of the dataset directories, which are at the same level as the script.
    datasets_to_check = ['laptop_14', 'restaurants_14', 'restaurants_15', 'restaurants_16']
    
    print("Searching for validation files in sibling directories...")
    
    files_to_upload = []
    for dataset_dir in datasets_to_check:
        local_path = Path(dataset_dir) / "data_validation.csv"
        if local_path.exists():
            files_to_upload.append(local_path)
            print(f"  ✅ Found: {local_path}")
        else:
            print(f"  ❌ Not found: {local_path}")

    if not files_to_upload:
        print("\nCould not find any validation files. Please ensure the directories exist.")
        return
        
    print(f"\nFound {len(files_to_upload)} validation file(s). Uploading to volume...")
    for path in files_to_upload:
        # The remote path on the volume will be the same as the local path
        # e.g., 'laptop_14/data_validation.csv'
        remote_path_str = str(path)
        
        print(f" - Uploading '{path}' to volume at path: '{remote_path_str}'")
        with open(path, "rb") as f:
            volume.write_file(remote_path_str, f)

    print("\n🚀 Launching remote model evaluation job on Modal...")
    test_model.remote()