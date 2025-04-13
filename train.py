from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import os

hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
model_id = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")


# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
# Patch rope_scaling for compatibility

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
    rope_scaling = config.rope_scaling
    config.rope_scaling = {
        "type": "linear",  # or "dynamic" if needed
        "factor": rope_scaling.get("factor", 1.0)
    }
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if not tokenizer.pad_token:
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
if not model.config.pad_token_id:
    model.config.pad_token_id = tokenizer.pad_token_id

# Load dataset from S3 (mapped to /opt/ml/input/data/train)
dataset = load_dataset("csv", data_files="/opt/ml/input/data/train/tweet_data.csv")
dataset = dataset["train"].select(range(50))  # just 50 rows
dataset = dataset.train_test_split(test_size=0.1)


# Prompt engineering
def format_text_template(example):
    instruction = """You are a famous Hollywood male actor preparing for a role in a movie where you will be playing Donald Trump. You will be asked questions and need to reply as Donald Trump would."""
    chat_template = [
        {"role": "system", "content": instruction},
        {"role": "actor", "content": example["tweet_text"]},
    ]
    example["text"] = tokenizer.apply_chat_template(chat_template, tokenize=False)
    return example

dataset = dataset.map(format_text_template)

# LoRA setup
modules = ["up_proj", "v_proj", "gate_proj", "q_proj", "k_proj", "down_proj", "o_proj"]
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)

model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="/opt/ml/model",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    max_steps=10,                    # <-- train for only 10 steps
    eval_strategy="steps",
    eval_steps=5,                    # <-- one quick eval halfway
    logging_steps=1,
    warmup_steps=0,                  # <-- skip warmup for quick test
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    deepspeed="ds_config.json",
    save_strategy="steps",
    save_steps=10,                   # <-- save once
    save_total_limit=1,              # <-- keep it light
    report_to="none"                 # <-- disable wandb for test
)


trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"].train_test_split(test_size=0.05)["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments
)

trainer.train()
