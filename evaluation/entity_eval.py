def locality_detected(locality, transcript):

    locality = locality.lower()
    transcript = transcript.lower()

    return locality in transcript