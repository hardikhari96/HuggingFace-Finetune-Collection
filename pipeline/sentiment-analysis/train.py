from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

dataset = load_dataset('csv', data_files='pipeline/sentiment-analysis/training_data.csv')

label_mapping = {"positive": 0, "neutral": 1, "negative": 2}

def map_labels(example):
    stripped_label = example['label'].strip()
    if stripped_label in label_mapping:
        example['label'] = label_mapping[stripped_label]
    else:
        raise ValueError(f"Unexpected label: '{stripped_label}'")
    return example

dataset = dataset['train'].map(map_labels)

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
valid_dataset = train_test_split['test']

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

train_dataset = train_dataset.map(preprocess_data, batched=True)
valid_dataset = valid_dataset.map(preprocess_data, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", num_labels=3)

training_args = TrainingArguments(
    output_dir="fine_tuned_models/sentiment-analysis/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='fine_tuned_models/sentiment-analysis/logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)
trainer.train()

model.save_pretrained("fine_tuned_models/sentiment-analysis/model")
tokenizer.save_pretrained("fine_tuned_models/sentiment-analysis/model")

fine_tuned_pipeline = pipeline("sentiment-analysis", model="fine_tuned_models/sentiment-analysis/model")

test_data = ["I love you", "I hate you", "You don't know you can do this"]
predictions = fine_tuned_pipeline(test_data)
print(predictions)
