from functools import cache
import torch
import numpy as np
from datasets import load_dataset, load_metric, Audio
import re, random, json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from DataCollator import DataCollatorCTCWithPadding
import wandb

# Loading dataset

########### UASpeech ############
dataset = load_dataset("json", data_files={
    "train": "../Manifests/UASpeech-train-manifest.json",
    "test": "../Manifests/UASpeech-test-manifest.json"
}, cache_dir="/l/users/karima.kadaoui/.cache/")
print("Dataset: UASpeech")

########### VCTK ############
# dataset = load_dataset("vctk", cache_dir="/l/users/karima.kadaoui/.cache/datasets/")
# # Changing sampling rate from 48K to 16K
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
# # Splitting dataset
# dataset = dataset["train"]
# dataset = dataset.train_test_split(test_size=0.1)

########### TIMIT ############
# dataset = load_dataset("timit_asr", cache_dir="/l/users/karima.kadaoui/.cache/datasets/")

print("Loaded dataset")
print(dataset)



# Remove special characters & lower case cuz CTC outputs letters not words
chars_to_ignore = '[\,\?\.\!\-\;\:\"]'
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore, '', batch["text"]).lower()
    return batch

dataset = dataset.map(remove_special_characters)
print("Removed special characters")


# Printing random transcriptions from the dataset
def show_random_elements(dataset, num_examples=10):
    if (num_examples >= len(dataset)): num_examples = len(dataset)
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks: # In case we've already picked that sentence before
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
        print("Random transcriptions:", dataset[pick])

show_random_elements(dataset["train"]["text"])


# Extract characters from dataset to construct vocab
def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text)) # Converts {s, e, t} to [l, i, s, t]. Set removes duplicates
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=uaspeech.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0])) #Union of letters b/w sets [0] because we have a nested list [vocab]
vocab_dict = {char: i for i, char in enumerate(vocab_list)}


# Replacing " " with more visible character |
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]


# Adding unknown token for OOV chars and blank token needed for CTC alignment
vocab_dict["[UNK]"] = len(vocab_dict) 
vocab_dict["[PAD]"] = len(vocab_dict)
# print(len(vocab_dict)) # Dim of linear layer that will be added on top of the pretrained checkpoin


# Dumping vocab into json
with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)
print("Extracted vocabulary to json")

# Instantiating tokenizer specifying vocab and unk, pad & word delimiter chars
tokenizer = Wav2Vec2CTCTokenizer(
    "../Vocabs/vocab.json", 
    unk_token="[UNK]", 
    pad_token="[PAD]", 
    word_delimiter_token="|"
)
print("Instantiated Tokenizer")


# Instantiating Feature Extractor.
# attention_mask should be used in general to mask padded tokens but Wav2Vec2's base checkpt has apparently better results without
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, 
    sampling_rate=16000, 
    padding_value=0.0, 
    do_normalize=True, 
    return_attention_mask=False
)
print("Instantiated Feature Extractor")


# Wrapping feature extractor and tokenizer into one processor entity
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, 
    tokenizer=tokenizer
)
print("Instantiated Processor")


# Printing random shape of speech, transcription and sr
rand_int = random.randint(0, len(dataset["train"]))
print("Random audio:")
print("Target text:", dataset["train"][rand_int]["text"])
print("Input array shape:", np.asarray(dataset["train"][rand_int]["audio"]["array"]).shape)
print("Sampling rate:", dataset["train"][rand_int]["audio"]["sampling_rate"])


# Extract features & encode labels
def prepare_dataset(batch):
    # Load & resample audio data
    audio = batch["audio"] 

    # Extract input_values from loaded audio. Processor only normalizes the data. In other models this can include more complex ftr extraction e.g. Log-mel ftr extraction
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    # Encode transcriptions to label ids
    with processor.as_target_processor(): # Calling processor redirects to the extractor. Adding as_target_processor redirects to tokenizer
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
print("Extracted features & encoded labels")

print("Label of random audio:", dataset["train"][rand_int]["labels"])


# Instantiating the data collator
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
print("Instantiated data collator")


# Loading metric
wer_metric = load_metric("wer")
print("Loaded metric:", wer_metric.name)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replacing the -100 collator padding by the PAD token
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids) # Pred
    # "We don't want to group tokens when computing the metrics" ???
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False) #GT

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Instantiating model. To save GPU memory, gradient_checkpointing is enabled and loss reduction set to mean
model = Wav2Vec2ForCTC.from_pretrained(
    # "facebook/wav2vec2-base-960h",
    "../Checkpoints/wav2vec2_timit+vctk+uasp_500warmup/checkpoint-14000/",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
).to("cuda")
print("Instantiated model: Wav2Vec2 CTC")

# First component of model: stack of CNNs used to extract acoustical features (independent of context)
# -> Sufficiently trained from pretext task & doesn't need to be fine-tuned
# SHOULD I FINE TUNE IT TOO SINCE THIS IS DYSARTHRIC DATA???
model.freeze_feature_encoder()
print("Froze feature encoder")
# print("FEATURE ENCODER NOT FROZEN")

wandb.watch(model, log="all", log_graph=(True))


# Specifying training arguments
training_args = TrainingArguments(
    output_dir="../Checkpoints/New",
    group_by_length=True,
    per_device_train_batch_size=16,
    # per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    num_train_epochs=7,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4, 
    weight_decay=0.005,
    warmup_steps=1000, 
    save_total_limit=2,
    push_to_hub=False,
    run_name="test-hp_warmup"
)
print("Specified training arguments")
print(training_args)


# Instantiating trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor
)
print("Trainer instantiated")

print("Start training:")
trainer.train()


# Inference and WER:

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
  return batch

results = dataset["test"].map(map_to_result, remove_columns=dataset["test"].column_names)

for i in range(len(results)):
    print("Text:", results[i]["text"], "        Predicted:", results[i]["pred_str"], "MATCHING" if results[i]["text"] == results[i]["pred_str"] else "")

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
