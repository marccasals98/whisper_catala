import evaluate


def compute_metrics(pred, tokenizer):
    """
    Computes the WER (Word Error Rate)

    Arguments:
    ----------
    pred : TODO
        Object that contains both the prediction and the ground truth
    tokenizer : TODO
        The tokenizer necessary to do the calculation
    
    Returns:
    --------
    wer: float
        The resulting metric. 
    """
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}