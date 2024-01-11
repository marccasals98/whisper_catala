from transformers import WhisperForConditionalGeneration
from config import MODEL

def get_model():
    """
    Imports the model

    Arguments:
    ----------
    None 

    Returns:
    --------
    model: 
        The Whisper model with the desired size. 
    """
    model = WhisperForConditionalGeneration.from_pretrained(MODEL['name'])

    # No tokens are forced as decoder outputs.
    model.config.forced_decoder_ids = None 

    # No tokens are suppressed during the generation.
    model.config.suppress_tokens = []

    return model

