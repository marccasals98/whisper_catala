from transformers import Seq2SeqTrainingArguments

MODEL = {
    'name': "openai/whisper-small",
    'language': "Catalan",
    'task': "transcribe"
}

DATASET =   {
    'name': "mozilla-foundation/common_voice_11_0",
    'language': "ca"
}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ca",  # change to a repo name of your choice
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
