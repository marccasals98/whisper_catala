from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from src.config import MODEL

# Pads / truncates the audio inputs to 30s and log-mel spectrogram
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL['name'])

# Load tokenizer 
tokenizer = WhisperTokenizer.from_pretrained(MODEL['name'], language=MODEL['language'], task=MODEL['task'])

