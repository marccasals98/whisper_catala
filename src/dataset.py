"""
DATASET: 

In this file we specify and treat the different datasets that we are using. 
"""

from datasets import load_dataset, DatasetDict
from datasets import Audio
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from config import MODEL, DATASET
import datasets
from datasets import IterableDatasetDict




def prepare_dataset(batch: datasets.formatting.formatting.LazyRow):
    """
    Readies the data for the model.

    Arguments:
    ----------
    batch: `datasets.formatting.formatting.LazyRow`
        A batch is a set of samples.

    Returns:
    --------
    batch : `datasets.formatting.formatting.LazyRow `
        A batch is a set of samples. We returned processed.
    """

    # Pads / truncates the audio inputs to 30s and log-mel spectrogram
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL['name'])

    # Load tokenizer 
    tokenizer = WhisperTokenizer.from_pretrained(MODEL['name'], language=MODEL['language'], task=MODEL['task'])
    
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch


def get_common_voice():
    """
    Do necessary preprocessing steps to obtain the samples of common_voice dataset

    Arguments:
    ----------
    None

    Returns: 
    --------
    common_voice: dict
        The Datset completed. 
    """
    # Treating the dataset as a dictionary
    common_voice = IterableDatasetDict()

    # Making the partition train/test: 
    # Before: split="train+validation"
    # Now : split="train"
    # streaming=True doesn't support this.
    # https://github.com/huggingface/datasets/issues/4804
    common_voice["train"] = load_dataset(DATASET['name'], DATASET['language'], split="train", use_auth_token=True, streaming=True)
    common_voice["test"] = load_dataset(DATASET['name'], DATASET['language'], split="test", use_auth_token=True, streaming=True)

    # For the moment we will discard some audio information to simplify the task. 
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    # Downsample on the fly:
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # Apply the function 'prepare_dataset' to all of the elements
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

    print("El tipus de common voice", type(common_voice))
    return common_voice

