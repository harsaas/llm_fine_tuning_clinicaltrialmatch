# 1. Transform raw text data into ChatML dataset splits
python src/format_data.py

# 2. Run the Unsloth fine-tuning execution pipeline
python src/train.py

# 3. Test your hybrid RAG query pattern with your local Qdrant database
python src/retrieve.py

# 4. Generate your validation analytics and save the dashboard visualization
python evaluation/evaluate_model.py


ingest_trials. py -> Search Engine Indexer 
--> It prepares the clinical trial protocols so they can be searched quickly.
--> This script does not train the LLM. It builds the library that allows your system to instantly pull up relevant trial data based on semantic meaning whenever a user pastes a patient's medical note
--> chops it into tiny 500-character pieces (chunks) using LangChain, and converts those pieces into mathematical vectors
(.venv-1) PS C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning> & c:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning\.venv-1\Scripts\python.exe c:/Users/harip/OneDrive/Desktop/llm_fine_tuning/Trial_Match_llm_finetuning/src/ingest_trials.py
Loading dataset from Hugging Face...
README.md: 13.4kB [00:00, ?B/s]
C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning\.venv-1\lib\site-packages\huggingface_hub\file_download.py:138: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\harip\.cache\huggingface\hub\datasets--louisbrulenaudet--clinical-trials. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Successfully loaded 1000 trial records.
Initializing embedding model...
Loading weights: 100%|██████████████████████████████████| 103/103 [00:00<00:00, 3372.65it/s]
Initializing local Qdrant store...
c:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning\src\ingest_trials.py:37: DeprecationWarning: `recreate_collection` method is deprecated and will be removed in the future. Use `collection_exists` to check collection existence and `create_collection` instead.
  client.recreate_collection(
Processing and embedding eligibility criteria...
Embedding trials: 100%|█████████████████████████████████| 1000/1000 [02:27<00:00,  6.80it/s]
✅ Ingestion complete! Vector DB at: trial_store.db
(.venv-1) PS C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning> 

format_data.py -> Fine tuning pipeline
-->  prepares conversation-style data to teach the LLM how to think like a medical screening expert.
--> It reads the raw trials and uses a template to generate synthetic data: a mock Patient profile, a Trial protocol, and a clean Verdict(Match or No Match). 
--> create train and test to avoid structured medical analysis without hallucinating.


(.venv-1) PS C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning> & c:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning\.venv-1\Scripts\python.exe c:/Users/harip/OneDrive/Desktop/llm_fine_tuning/Trial_Match_llm_finetuning/src/format_data.py
Loading raw datasets...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loaded 500 trials
Wrote 800 train samples to data\train.jsonl
Wrote 200 val samples to data\val.jsonl
'[WinError 10038] An operation was attempted on something that is not a socket' thrown while requesting GET https://huggingface.co/datasets/louisbrulenaudet/clinical-trials/resolve/67359a8b9906e52190457a4be6bc9bce903d7e23/data/train-00000-of-00008.parquet
Retrying in 1s [Retry 1/5].
(.venv-1) PS C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning>


--Train script -----------

Loading model weights via Unsloth...
==((====))==  Unsloth 2026.5.8: Fast Llama patching. Transformers: 5.5.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights: 100%
 254/254 [00:01<00:00,  1.70s/it]
Unsloth: Will load unsloth/Llama-3.2-3B-Instruct-bnb-4bit as a legacy tokenizer.
Parsing dataset files and processing numeric token matrices...
Map: 100%
 800/800 [00:01<00:00, 547.03 examples/s]
Map: 100%
 200/200 [00:00<00:00, 360.35 examples/s]
Initializing pure standard Hugging Face Trainer pipeline...
🚀 Executing training optimization steps...
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 800 | Num Epochs = 2 | Total steps = 60
O^O/ \_/ \    Batch size per device = 4 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (4 x 4 x 1) = 16
 "-____-"     Trainable parameters = 24,313,856 of 3,237,063,680 (0.75% trained)
`use_return_dict` is deprecated! Use `return_dict` instead!
Unsloth: Will smartly offload gradients to save VRAM!
Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.
 [60/60 09:26, Epoch 1/2]
Step	Training Loss	Validation Loss
10	2.361331	2.238496
20	1.537237	1.518947
30	1.494586	1.442314
40	1.500848	1.419743
50	1.528578	1.404930
60	1.568208	1.399392
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:71: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:281: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:172: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:202: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:254: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:71: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:281: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:172: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:202: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/modeling_attn_mask_utils.py:254: FutureWarning: The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`.
  warnings.warn(DEPRECATION_MESSAGE, FutureWarning)
Unsloth: Restored added_tokens_decoder metadata in outputs/checkpoint-60/tokenizer_config.json.
Unsloth: Restored added_tokens_decoder metadata in lora_model/tokenizer_config.json.
✅ Optimization finished! 'lora_model' folder created successfully.

Evaluation metrics :

outstanding results from a real validation run on a 3-Billion parameter model! Your fine-tuned model achieved a highly competitive 81.50% Accuracy and an exceptional 0.8800 Precision Score, meaning that when it flags a patient as a match, it is correct nearly 90% of the time. This solves a massive business pain point by cutting down on hours wasted reviewing bad clinical leads

============================================================
📊 PRODUCTION LIVE MODEL PERFORMANCE REPORT
============================================================
   • Total Validation Samples Evaluated: 200
   • True Positives (TP):                66
   • False Positives (FP):               9
   • False Negatives (FN):               28
   • True Negatives (TN):                97
------------------------------------------------------------
🟩 CALCULATED METRICS:
   • Model Accuracy:                    81.50%
   • Precision Score:                   0.8800
   • Recall Score (Sensitivity):        0.7021
   • F2-Score (Clinical Target Focus):  0.7317
============================================================

--> Fit the colab eval data in to the evaluate the colab vs base lineversion to check the improvement in performance:

(.venv-1) PS C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning> & c:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning\.venv-1\Scripts\python.exe c:/Users/harip/OneDrive/Desktop/llm_fine_tuning/Trial_Match_llm_finetuning/evaluation/evaluation_fit_param_colab.py
🧬 Initializing Text-Based Performance Metrics Suite...

============================================================
📊 TRIALMATCH-LLM FOUNDATIONAL METRICS REPORT
============================================================
🔹 BASE MODEL (Llama-3.2-3B Baseline Heuristics):
   • Precision:                 0.5507
   • Recall (Sensitivity):      0.4043
   • F2-Score (Recall Focus):   0.4270
------------------------------------------------------------
🟩 FINE-TUNED MODEL (TrialMatch-LLM Dynamic Performance):
   • Total Samples Evaluated:   200
   • Model Accuracy:            81.50%
   • Precision:                 0.8800
   • Recall (Sensitivity):      0.7021
   • F2-Score (Recall Focus):   0.7317
============================================================
📈 PRODUCTION DELTA IMPROVEMENT PROFILE:
   • Precision Lift:           +32.9%
   • Recall Lift:              +29.8%
   • F2-Score Optimization:     +30.5%
============================================================

(.venv-1) PS C:\Users\harip\OneDrive\Desktop\llm_fine_tuning\Trial_Match_llm_finetuning> 

POSTED THE MODEL IN HUGGING FACE 

https://huggingface.co/harsaas/TrialMatch-LLM-Qwen1.5B
