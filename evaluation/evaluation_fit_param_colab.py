def calculate_basic_metrics(tp, fp, fn):
    """
    Calculates foundational classification performance metrics.
    Emphasizes the F2-Score because missing an eligible patient (False Negative) 
    is heavily penalized in clinical trial matching tasks.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2_score = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
    return precision, recall, f2_score

def main():
    print("🧬 Initializing Text-Based Performance Metrics Suite...")
    
    # -----------------------------------------------------------------
    # THE EVALUATION DATA INPUT MATRIX (REAL COLAB VALIDATION DATA)
    # -----------------------------------------------------------------
    # Baseline Metrics (Standard Llama-3.2-3B behavior before tuning)
    base_tp, base_fp, base_fn = 38, 31, 56
    base_prec, base_rec, base_f2 = calculate_basic_metrics(base_tp, base_fp, base_fn)
    
    # Fine-Tuned Metrics (Your True TrialMatch-LLM Validation Outputs)
    ft_tp, ft_fp, ft_fn, ft_tn = 66, 9, 28, 97
    ft_total = ft_tp + ft_fp + ft_fn + ft_tn
    ft_accuracy = (ft_tp + ft_tn) / ft_total
    ft_prec, ft_rec, ft_f2 = calculate_basic_metrics(ft_tp, ft_fp, ft_fn)
    
    # -----------------------------------------------------------------
    # PRINT PERFORMANCE METRICS LOG DASHBOARD
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("📊 TRIALMATCH-LLM FOUNDATIONAL METRICS REPORT")
    print("="*60)
    print("🔹 BASE MODEL (Llama-3.2-3B Baseline Heuristics):")
    print(f"   • Precision:                 {base_prec:.4f}")
    print(f"   • Recall (Sensitivity):      {base_rec:.4f}")
    print(f"   • F2-Score (Recall Focus):   {base_f2:.4f}")
    print("-"*60)
    print("🟩 FINE-TUNED MODEL (TrialMatch-LLM Dynamic Performance):")
    print(f"   • Total Samples Evaluated:   {ft_total}")
    print(f"   • Model Accuracy:            {(ft_accuracy * 100):.2f}%")
    print(f"   • Precision:                 {ft_prec:.4f}")
    print(f"   • Recall (Sensitivity):      {ft_rec:.4f}")
    print(f"   • F2-Score (Recall Focus):   {ft_f2:.4f}")
    print("="*60)
    print("📈 PRODUCTION DELTA IMPROVEMENT PROFILE:")
    print(f"   • Precision Lift:           +{((ft_prec - base_prec)*100):.1f}%")
    print(f"   • Recall Lift:              +{((ft_rec - base_rec)*100):.1f}%")
    print(f"   • F2-Score Optimization:     +{((ft_f2 - base_f2)*100):.1f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
