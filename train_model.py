from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback
import tensorflow as tf
import numpy as np
import evaluate

# IMDb datasetini yükle
dataset = load_dataset("imdb")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)

# Tokenize edilmiş dataset
tokenized = dataset.map(preprocess, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# Accuracy metriği
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Label mapping
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Eğitim parametreleri
batch_size = 16
epochs = 2
steps = (len(tokenized["train"]) // batch_size) * epochs
optimizer, _ = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0, 
    num_train_steps=steps
)

# Model (PyTorch → TensorFlow dönüşümü ile)
model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id,
    from_pt=True  # HATAYI ÖNLEMEK İÇİN EKLENDİ
)

# TensorFlow dataset hazırla
train_set = model.prepare_tf_dataset(
    tokenized["train"], shuffle=True, batch_size=batch_size, collate_fn=collator
)
val_set = model.prepare_tf_dataset(
    tokenized["test"], shuffle=False, batch_size=batch_size, collate_fn=collator
)

# Model derleme
model.compile(optimizer=optimizer)

# Callback (accuracy ölçümü için)
metric_cb = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=val_set)

# Model eğitimi
model.fit(train_set, validation_data=val_set, epochs=epochs, callbacks=[metric_cb])

# Api yazdıktan sonra model ve tokenizer kaydet
# model.save_pretrained("imdb_model")
# tokenizer.save_pretrained("imdb_model")
