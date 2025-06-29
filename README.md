# Aspect-Based Sentiment Analysis with Llama 3

This repository contains experiments on aspect-based sentiment analysis (ABSA) using Meta's Llama 3 (3-8B, 3-70B, 3.1-70B, 3.3-8B) series language models.  The project covers data preparation, fine-tuning with LoRA adapters, zero‑shot/few‑shot evaluation and model testing.  All training and inference jobs are configured to run on [Modal](https://modal.com/) GPU containers.

## Repository Structure

```
finetuning(temperature)/
├── datasetCreation/        # Create a JSONL training set from CSV files
├── downloadModel/          # Fetch base model weights into a Modal volume
├── modelTraining/          # LoRA fine-tuning script
├── modelTrainingInstruct/  # LoRA fine-tuning with instruction prompts
└── modelTesting/           # Evaluate the fine-tuned adapter

zeroshot:fewshot(temperature)/
├── data/                   # Raw XML datasets and converted CSVs
└── main.ipynb              # Preprocessing and zero/few-shot experiments
```

## Techniques and Tools Used

- **Meta Llama 3 models** – Experiments use models hosted on the Hugging Face Hub.
- **LoRA (Low-Rank Adaptation)** via the [PEFT](https://github.com/huggingface/peft) library for parameter‑efficient fine-tuning.
- **Modal** – All heavy jobs (model download, training and testing) run remotely on Modal GPU instances and use a network file system volume for persistence.
- **Datasets** – SemEval 2014/2015/2016 ABSA datasets are converted from XML to CSV for model evaluation. A custom instruction dataset is assembled into JSONL for fine‑tuning.
- **Python libraries** – `transformers`, `torch`, `datasets`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `scikit-learn`, and `tensorboard`.

## Setup

### Prerequisites

- Python 3.10
- A Modal account with GPU access
- A Hugging Face account with permission to download the Llama models

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Nastya
   ```
2. **Install the dependencies**
   ```bash
   pip install modal transformers torch datasets peft trl bitsandbytes accelerate scikit-learn tensorboard
   ```
3. **Configure Hugging Face and Modal credentials**
   - Create a secret named `my-huggingface-secret` in your Modal account containing your Hugging Face token (`HF_TOKEN`).  One way is:
     ```bash
     modal secret put my-huggingface-secret HF_TOKEN=your-hf-token
     ```
   - Ensure you have access to the Llama models you plan to use.

### First-Time Modal Setup

If you have never used Modal before, authenticate and create the storage volume that all scripts rely on:

```bash
modal token new                     # open a browser to log in
modal volume create llama-models    # persistent storage for model weights
```

Once authenticated, you can run `modal run` commands from this repository.

## How to Run the Scripts

### 1. Prepare the Dataset

```
cd finetuning(temperature)/datasetCreation
python datasetCreation.py
```

This script merges the CSV files in the `datasetCreation` folder and produces `llama_finetuning_instruct_data.jsonl` used for training.

### 2. Download the Base Model

```
cd ../downloadModel
modal run downloadModel.py
```

The script downloads the selected Llama 3 base model into a Modal volume so that training jobs can access it quickly.

### 3. Fine‑Tune with LoRA

Two training scripts are provided.  `modelTraining/trainModel.py` fine‑tunes on generic prompts, while `modelTrainingInstruct/trainModelInstruct.py` uses instruction‑style prompts.

```
cd ../modelTraining
modal run trainModel.py
```

The fine‑tuned adapter and tokenizer are saved to the `model_output` path in the Modal volume.

### 4. Evaluate the Adapter

```
cd ../modelTesting
modal run testModel.py
```

This job loads the base model and the LoRA adapter, merges them for inference and evaluates accuracy and F1 score on the four validation datasets (`laptop_14`, `restaurants_14`, `restaurants_15`, `restaurants_16`).

### 5. Zero‑Shot/Few‑Shot Experiments

The notebook in `zeroshot:fewshot(temperature)/main.ipynb` demonstrates XML → CSV conversion and evaluation of public Llama models via the Hugging Face Inference API.  Run the notebook in Jupyter and supply your Hugging Face token when prompted.

## Notes

- Modal automatically installs dependencies declared in each script's `modal.Image` specification.
- The evaluation datasets are located under `finetuning(temperature)/modelTesting/`.
- Replace `YOUR HUGGINGFACE TOKEN` in the notebook with a real token to perform zero/few-shot calls.

## License

This project is provided for educational purposes.  The underlying datasets and models may have their own licenses and usage restrictions.  Ensure you comply with Meta's and Hugging Face's terms of service when using the Llama models.
