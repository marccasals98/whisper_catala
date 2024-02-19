from transformers import WhisperForConditionalGeneration
from config import MODEL

def get_model()->WhisperForConditionalGeneration:
    """
    Imports the model

    Arguments:
    ----------
    None 

    Returns:
    --------
    model: `WhisperForConditionalGeneration`
        The Whisper model with the desired size. 
    """
    model = WhisperForConditionalGeneration.from_pretrained(MODEL['name'], local_files_only=MODEL['local_files_only'])

    # No tokens are forced as decoder outputs.
    model.config.forced_decoder_ids = None 

    # No tokens are suppressed during the generation.
    model.config.suppress_tokens = []
    print("El tipus de model", type(model))
    return model

