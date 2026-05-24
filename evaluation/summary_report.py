import pandas as pd


def generate_summary(df):

    summary = (
        df.groupby("model")
        .agg({
            "wer": "mean",
            "cer": "mean",
            "latency": "mean",
            "entity_correct": "mean",
            "entity_score": "mean"
        })
        .reset_index()
    )

    summary["wer"] = summary["wer"].round(3)
    summary["cer"] = summary["cer"].round(3)
    summary["latency"] = summary["latency"].round(2)
    summary["entity_correct"] = (
        summary["entity_correct"] * 100
    ).round(1)

    summary["entity_score"] = (
        summary["entity_score"]
    ).round(1)

    return summary