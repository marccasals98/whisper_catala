from transformers import Seq2SeqTrainer
from config import training_args
from model import get_model
from metrics import compute_metrics 
from dataset import get_common_voice
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperProcessor
from config import MODEL

def get_trainer():
    """
    Gets the Seq2SeqTrainer

    Inputs:
    -------
    None

    Returns:
    --------
    trainer: Seq2SeqTrainer
        The trainer that is created with all hyperparameters.
    """
    
    processor = WhisperProcessor.from_pretrained(MODEL['name'], language=MODEL['language'], task=MODEL['task'])
    
    common_voice = get_common_voice()
    
    model = get_model()


    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    return trainer