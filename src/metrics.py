import evaluate
from transformers import WhisperTokenizer
from config import MODEL



def compute_metrics(pred):
    """
    Computes the WER (Word Error Rate). The word error rate is 
    
    WER=(S+B+I)/N,

    where

    * S = subtitutions
    * B = deletions
    * I = insertions
    * N = number of total words.

    Arguments:
    ----------
    pred : TODO
        Object that contains both the prediction and the ground truth
    
    Returns:
    --------
    wer: float
        The resulting metric. 

    IMPORTANT: This function will passed as argument to the function Seq2SeqTrainer. So it requires to not have
    any other argument appart from pred. 

    """
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Load tokenizer 
    tokenizer = WhisperTokenizer.from_pretrained(MODEL['name'], language=MODEL['language'], task=MODEL['task'])

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute  (predictions=pred_str, references=label_str)

    return {"wer": wer}