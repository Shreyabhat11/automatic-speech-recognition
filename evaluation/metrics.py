from jiwer import wer, cer

def calculate_metrics(reference, hypothesis):

    word_error = wer(reference, hypothesis)
    char_error = cer(reference, hypothesis)

    return {
        "WER": word_error,
        "CER": char_error
    }