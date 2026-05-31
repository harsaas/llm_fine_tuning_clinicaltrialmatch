import os
import torch

# 1. Setup Architecture Configs
max_seq_length = 2048   
dtype = None            
load_in_4bit = True     

# Load unsloth first to apply internal system memory optimizations
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

TINY_RUN = False  # Set to False to run your production 60-step tuning cycle

train_file = "train.jsonl" if os.path.exists("train.jsonl") else os.path.join("data", "train.jsonl")
val_file = "val.jsonl" if os.path.exists("val.jsonl") else os.path.join("data", "val.jsonl")

max_steps = 2 if TINY_RUN else 60
max_train_samples = 8 if TINY_RUN else None
max_val_samples = 4 if TINY_RUN else None

print("Loading model weights via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
tokenizer.padding_side = 'right'

# =====================================================================
# 💡 THE FIX: CONVERT RAW CHAT TEXT INTO MATH VECTORS IMMEDIATELY
# =====================================================================
def tokenize_and_format_func(examples):
    convos = examples["messages"]
    
    # 1. Map text onto ChatML structural constraints
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    
    # 2. Tokenize text strings directly into input_ids and attention_masks
    tokenized = tokenizer(texts, truncation=True, max_length=max_seq_length)
    
    # 3. For Causal Language Models, labels are identical duplicates of input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Parsing dataset files and processing numeric token matrices...")
train_ds = load_dataset("json", data_files=train_file, split="train")
val_ds = load_dataset("json", data_files=val_file, split="train")

if max_train_samples is not None:
    train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
if max_val_samples is not None:
    val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))

# Run the unified math token mapping function across the shards
# We remove raw un-encoded columns so the trainer only sees numerical vectors
train_ds = train_ds.map(tokenize_and_format_func, batched=True, remove_columns=["messages"])
val_ds = val_ds.map(tokenize_and_format_func, batched=True, remove_columns=["messages"])

# 2. Configure Standard Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,      
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=max_steps,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    eval_strategy="steps",   
    eval_steps=10,           
    output_dir="outputs",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    remove_unused_columns=True, # Can safely keep this True now that columns are named correctly
)

# A data collator dynamically pads variable token sequences in a batch to the same length
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# =====================================================================
# 3. INITIALIZE STANDARD HUGGING FACE TRAINER
# =====================================================================
print("Initializing pure standard Hugging Face Trainer pipeline...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

print("🚀 Executing training optimization steps...")
trainer.train()

# Save final adapter weights locally to your Colab session
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("✅ Optimization finished! 'lora_model' folder created successfully.")
