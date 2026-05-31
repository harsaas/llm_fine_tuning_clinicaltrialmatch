import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

# 1. Load your fine-tuned LoRA adapters in Colab's GPU
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # Points to your freshly trained folder
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Optimize for 2x faster evaluation generation

# 2. Initialize Confusion Matrix Counters
tp, fp, fn, tn = 0, 0, 0, 0

print("🧬 Evaluating fine-tuned model against val.jsonl...")

# 3. Read your validation dataset file line-by-line
with open("val.jsonl", "r") as f:
    for line in tqdm(f, desc="Processing validation records"):
        data = json.loads(line)
        messages = data["messages"]
        
        # Extract the user prompt and the true ground-truth answer
        user_prompt = messages[1]["content"]
        true_answer = messages[2]["content"].upper() # E.g., Contains "VERDICT: MATCH" or "VERDICT: NO MATCH"
        
        # Format input tokens matching your ChatML sequence template
        inputs = tokenizer.apply_chat_template(
            [messages[0], messages[1]], # System prompt + User question
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")
        
        # Generate the model's prediction
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, max_new_tokens=32, temperature=0.1)
        
        # Decode the output text string
        predicted_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).upper()
        
        # Determine Ground-Truth and Predicted Boolean classes
        is_true_match = "VERDICT: MATCH" in true_answer
        is_pred_match = "VERDICT: MATCH" in predicted_text
        
        # 4. Populate Confusion Matrix Logic
        if is_true_match and is_pred_match:
            tp += 1
        elif not is_true_match and is_pred_match:
            fp += 1
        elif is_true_match and not is_pred_match:
            fn += 1
        else:
            tn += 1

# 5. Compute Foundational Mathematical Metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

# 6. Display Real-Time Metrics Output Panel
print("\n" + "="*60)
print("📊 PRODUCTION LIVE MODEL PERFORMANCE REPORT")
print("="*60)
print(f"   • Total Validation Samples Evaluated: {tp + fp + fn + tn}")
print(f"   • True Positives (TP):                {tp}")
print(f"   • False Positives (FP):               {fp}")
print(f"   • False Negatives (FN):               {fn}")
print(f"   • True Negatives (TN):                {tn}")
print("-"*60)
print(f"🟩 CALCULATED METRICS:")
print(f"   • Model Accuracy:                    {(accuracy * 100):.2f}%")
print(f"   • Precision Score:                   {precision:.4f}")
print(f"   • Recall Score (Sensitivity):        {recall:.4f}")
print(f"   • F2-Score (Clinical Target Focus):  {f2_score:.4f}")
print("="*60 + "\n")
