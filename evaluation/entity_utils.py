from rapidfuzz import fuzz

def fuzzy_entity_match(locality, transcript):

    locality = locality.lower().replace("_", " ")
    transcript = transcript.lower()

    score = fuzz.partial_ratio(
        locality,
        transcript
    )

    return score >= 70, score