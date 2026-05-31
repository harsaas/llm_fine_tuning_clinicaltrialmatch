# Parses raw trials, generates simple synthetic match/no-match chat samples,
# and writes train/val JSONL files under data/.

import json
import os
import random
import re

from datasets import load_dataset


def _mentions_exclusion(criteria_text: str, keyword_pattern: str) -> bool:
    """Heuristic: keyword appears near exclusion/negation language."""
    text = criteria_text.lower()
    markers = (
        "exclusion",
        "exclude",
        "excluded",
        "must not",
        "should not",
        "not eligible",
        "ineligible",
        "no ",
        "without ",
    )

    for match in re.finditer(keyword_pattern, text):
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        window = text[start:end]
        if any(m in window for m in markers):
            return True
    return False


def _extract_age_constraints(criteria_text: str) -> tuple[int | None, int | None]:
    """Very small heuristic age parser: returns (min_age, max_age) if detectable."""
    text = criteria_text.lower()

    def _valid_age(v: int) -> bool:
        return 0 < v <= 120

    # Range like: "50-85 years", "50 to 85 yrs"
    m = re.search(r"\b(\d{1,3})\s*(?:-|to)\s*(\d{1,3})\s*(?:years|yrs)\b", text)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        if _valid_age(lo) and _valid_age(hi) and lo <= hi:
            return lo, hi

    # Minimum age like:
    # - "Age >= 18"
    # - "Age at least 18"
    # - "18 years and older"
    m = re.search(r"\bage\b[^\n]{0,40}?(?:>=|≥|at\s+least)\s*(\d{1,3})\b", text)
    if m:
        v = int(m.group(1))
        if _valid_age(v):
            return v, None

    m = re.search(r"\b(\d{1,3})\s*(?:years|yrs)\s*(?:and\s+older|\+)\b", text)
    if m:
        v = int(m.group(1))
        if _valid_age(v):
            return v, None

    # Maximum age like:
    # - "Age <= 65"
    # - "Age up to 65"
    # - "up to 65 years"
    m = re.search(r"\bage\b[^\n]{0,40}?(?:<=|≤|up\s+to)\s*(\d{1,3})\b", text)
    if m:
        v = int(m.group(1))
        if _valid_age(v):
            return None, v

    m = re.search(r"\bup\s+to\s+(\d{1,3})\s*(?:years|yrs)\b", text)
    if m:
        v = int(m.group(1))
        if _valid_age(v):
            return None, v

    return None, None


def generate_synthetic_samples(trial: dict) -> list[dict]:
    nct_id = trial.get("id") or trial.get("nct_id") or "NCT_Unknown"
    title = trial.get("title", "Unknown title")
    criteria = str(trial.get("eligibility", trial.get("eligibility_criteria", "")))
    criteria_short = criteria[:800]

    match_system = "You are TrialMatch-LLM. Match the patient to the protocol."

    # Decide which contradictions to generate for NO MATCH.
    contradictions: list[str] = []

    min_age, max_age = _extract_age_constraints(criteria)
    if min_age is not None or max_age is not None:
        contradictions.append("age_outside_range")
    if _mentions_exclusion(criteria, r"\bpregnan(t|cy)\b"):
        contradictions.append("pregnancy")
    if _mentions_exclusion(criteria, r"\bdiabet(e|es|ic)\b"):
        contradictions.append("diabetes")
    if _mentions_exclusion(criteria, r"\b(cancer|tumou?r|malignan|neoplasm)\w*\b"):
        contradictions.append("cancer_or_tumor")

    # If none were detectable from criteria text, keep one generic contradiction.
    if not contradictions:
        contradictions.append("generic_contradiction")

    # MATCH patient profile: pick an age that fits constraints if detectable.
    match_age = 35
    if min_age is not None and max_age is not None:
        match_age = (min_age + max_age) // 2
    elif min_age is not None:
        match_age = max(min_age, 18 if min_age >= 18 else min_age)
    elif max_age is not None:
        match_age = min(35, max_age)

    match_ehr_lines = [
        f"Age: {match_age}",
        "Sex: Female",
        "Pregnancy: No",
        "Diabetes: No",
        "Cancer/Tumor history: No",
    ]
    match_user = (
        "=== PATIENT EHR ===\n"
        + "\n".join(match_ehr_lines)
        + f"\n\n=== PROTOCOL ({nct_id}) ===\n{criteria_short}"
    )
    match_assistant = (
        "VERDICT: MATCH\n"
        "RATIONALE:\n"
        "- Patient appears to meet the basic screening constraints provided.\n"
        "- No obvious exclusion triggers found in the patient summary."
    )

    # NO MATCH patient profile: inject contradictions.
    no_match_ehr_lines = [f"Age: {match_age}", "Sex: Female"]
    rationale_bullets: list[str] = []

    if "age_outside_range" in contradictions:
        if min_age is not None and min_age >= 18:
            # Adult-only trial -> make it clearly pediatric.
            bad_age = 16
            no_match_ehr_lines[0] = f"Age: {bad_age}"
            rationale_bullets.append(f"- Age below minimum (requires ≥ {min_age}).")
        elif min_age is not None and max_age is not None:
            bad_age = max_age + 5
            no_match_ehr_lines[0] = f"Age: {bad_age}"
            rationale_bullets.append(f"- Age outside allowed range (requires {min_age}-{max_age}).")
        elif min_age is not None:
            bad_age = max(1, min_age - 2)
            no_match_ehr_lines[0] = f"Age: {bad_age}"
            rationale_bullets.append(f"- Age below minimum (requires ≥ {min_age}).")
        elif max_age is not None:
            bad_age = max_age + 5
            no_match_ehr_lines[0] = f"Age: {bad_age}"
            rationale_bullets.append(f"- Age above maximum (requires ≤ {max_age}).")
    if "pregnancy" in contradictions:
        no_match_ehr_lines.append("Pregnancy: Yes")
        rationale_bullets.append("- Currently pregnant.")
    else:
        no_match_ehr_lines.append("Pregnancy: No")

    if "diabetes" in contradictions:
        no_match_ehr_lines.append("Diabetes: Yes (poorly controlled)")
        rationale_bullets.append("- Poorly controlled diabetes noted.")
    else:
        no_match_ehr_lines.append("Diabetes: No")

    if "cancer_or_tumor" in contradictions:
        no_match_ehr_lines.append("Cancer/Tumor history: Active malignancy")
        rationale_bullets.append("- Active cancer/tumor history reported.")
    else:
        no_match_ehr_lines.append("Cancer/Tumor history: No")

    if "generic_contradiction" in contradictions:
        no_match_ehr_lines.append("Other: Has an exclusionary comorbidity not permitted by protocol")
        rationale_bullets.append("- Patient has a protocol-conflicting comorbidity.")

    no_match_user = (
        "=== PATIENT EHR ===\n"
        + "\n".join(no_match_ehr_lines)
        + f"\n\n=== PROTOCOL ({nct_id}) ===\n{criteria_short}"
    )
    no_match_assistant = "VERDICT: NO MATCH\nRATIONALE:\n" + "\n".join(rationale_bullets)

    return [
        {
            "messages": [
                {"role": "system", "content": match_system},
                {"role": "user", "content": match_user},
                {"role": "assistant", "content": match_assistant},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": match_system},
                {"role": "user", "content": no_match_user},
                {"role": "assistant", "content": no_match_assistant},
            ]
        },
    ]


def prepare_pipeline_datasets(
    *,
    dataset_name: str = "louisbrulenaudet/clinical-trials",
    split: str = "train",
    max_trials: int = 500,
    out_dir: str = "data",
    seed: int = 3407,
) -> None:
    print("Loading raw datasets...")

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    raw_trials: list[dict] = []
    it = iter(dataset)
    for _ in range(max_trials):
        try:
            raw_trials.append(next(it))
        except StopIteration:
            break
        except Exception as exc:
            print(f"Warning: stopped early after {len(raw_trials)} trials due to dataset read error: {exc}")
            break
    print(f"Loaded {len(raw_trials)} trials")

    if not raw_trials:
        raise RuntimeError(
            "No trials were loaded. Check your internet connection, HF rate limits, or set HF_TOKEN."
        )

    all_chat_samples: list[dict] = []
    for trial in raw_trials:
        all_chat_samples.extend(generate_synthetic_samples(trial))

    random.seed(seed)
    random.shuffle(all_chat_samples)

    split_idx = int(0.8 * len(all_chat_samples))
    train_data = all_chat_samples[:split_idx]
    val_data = all_chat_samples[split_idx:]

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_data)} train samples to {train_path}")
    print(f"Wrote {len(val_data)} val samples to {val_path}")


if __name__ == "__main__":
    prepare_pipeline_datasets()
