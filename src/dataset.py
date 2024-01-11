"""
DATASET: 

In this file we specify and treat the different datasets that we are using. 
"""

from datasets import load_dataset, DatasetDict
from datasets import Audio


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Readies the data for the model.

    Arguments:
    ----------
    batch: type
        Each batch of data.
    feature_extractor: TODO
        Pads / truncates the audio inputs to 30s and log-mel spectrogram
    tokenizer: TODO    
        The tokenizer goes from tokens to strings.

    Returns:
    --------
    batch : type 
        Each batch of data
    """
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch




# Treating the dataset as a dictionary
common_voice = DatasetDict()

# Making the partition train/test: 
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)

# For the moment we will discard some audio information to simplify the task. 
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# Downsample on the fly:
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Apply the function 'prepare_dataset' to all of the elements
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

