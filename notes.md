# 1. Transform raw text data into ChatML dataset splits
python src/format_data.py

# 2. Run the Unsloth fine-tuning execution pipeline
python src/train.py

# 3. Test your hybrid RAG query pattern with your local Qdrant database
python src/retrieve.py

# 4. Generate your validation analytics and save the dashboard visualization
python evaluation/evaluate_model.py