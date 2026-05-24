from jiwer import wer
from jiwer import cer


def calculate_wer(
    ground_truth,
    transcript
):

    try:
        return round(
            wer(
                ground_truth.lower(),
                transcript.lower()
            ),
            3
        )

    except:
        return 1.0


def calculate_cer(
    ground_truth,
    transcript
):

    try:
        return round(
            cer(
                ground_truth.lower(),
                transcript.lower()
            ),
            3
        )

    except:
        return 1.0