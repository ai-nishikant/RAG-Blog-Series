from rouge_score import rouge_scorer

def rouge_scores(reference: str, output: str) -> dict:
    """
    Returns ROUGE-1 and ROUGE-L F1 scores rounded to 3 decimals.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, output)
    return {k: round(v.fmeasure, 3) for k, v in s.items()}
