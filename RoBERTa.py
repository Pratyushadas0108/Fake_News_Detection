import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
from transformers import (
    AutoTokenizer,
    RobertaModel,
    RobertaPreTrainedModel,
    Trainer,
    TrainingArguments
)
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ---------------- CONFIG ----------------
MODEL_NAME = "roberta-large"
NUM_MAIN_LABELS = 2
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 4
LR = 1.5e-5
GRAD_ACC = 4
WEIGHT_DECAY = 0.01

LABEL_MAP = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATA LOADING ----------------
SPEAKER_MAP, PARTY_MAP, SUBJECT_MAP = {}, {}, {}
NUM_SPEAKERS = NUM_PARTIES = NUM_SUBJECTS = 0

def load_data(path, is_train=False):
    global SPEAKER_MAP, PARTY_MAP, SUBJECT_MAP
    global NUM_SPEAKERS, NUM_PARTIES, NUM_SUBJECTS

    df = pd.read_csv(path).fillna("")

    df["label"] = df["label"].map(LABEL_MAP).astype(int)

    df["combined_text"] = (
        df["statement"] + " [SEP] " +
        df["context"] + " [SEP] " +
        df["justification"] + " [SEP] " +
        "Speaker: " + df["speaker"]
    )

    if is_train:
        SPEAKER_MAP = {v:i for i,v in enumerate(df["speaker"].unique())}
        PARTY_MAP = {v:i for i,v in enumerate(df["party"].unique())}
        SUBJECT_MAP = {v:i for i,v in enumerate(df["subject"].unique())}

        NUM_SPEAKERS = len(SPEAKER_MAP)
        NUM_PARTIES = len(PARTY_MAP)
        NUM_SUBJECTS = len(SUBJECT_MAP)

    return (
        df["combined_text"].tolist(),
        df["label"].tolist(),
        df["speaker"].map(SPEAKER_MAP).fillna(0).astype(int).tolist(),
        df["party"].map(PARTY_MAP).fillna(0).astype(int).tolist(),
        df["subject"].map(SUBJECT_MAP).fillna(0).astype(int).tolist()
    )

train_texts, train_labels, spk_tr, party_tr, subj_tr = load_data("train.csv", True)
val_texts, val_labels, spk_v, party_v, subj_v = load_data("valid.csv")
test_texts, test_labels, spk_te, party_te, subj_te = load_data("test.csv")

# ---------------- TOKENIZATION ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)

# ---------------- DATASET ----------------
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, enc, y, s, p, sub):
        self.enc, self.y, self.s, self.p, self.sub = enc, y, s, p, sub

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k,v in self.enc.items()}
        item["labels"] = torch.tensor(self.y[i])
        item["speaker_labels"] = torch.tensor(self.s[i])
        item["party_labels"] = torch.tensor(self.p[i])
        item["subject_labels"] = torch.tensor(self.sub[i])
        return item

    def __len__(self):
        return len(self.y)

train_ds = NewsDataset(tokenize(train_texts), train_labels, spk_tr, party_tr, subj_tr)
val_ds   = NewsDataset(tokenize(val_texts), val_labels, spk_v, party_v, subj_v)
test_ds  = NewsDataset(tokenize(test_texts), test_labels, spk_te, party_te, subj_te)

# ---------------- CLASS WEIGHTS ----------------
counts = Counter(train_labels)
weights = torch.tensor(
    [sum(counts.values()) / (NUM_MAIN_LABELS * counts[i]) for i in range(NUM_MAIN_LABELS)],
    device=device
)

# ---------------- MODEL ----------------
class RobertaMTL(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.main = nn.Linear(config.hidden_size, NUM_MAIN_LABELS)
        self.spk = nn.Linear(config.hidden_size, NUM_SPEAKERS)
        self.party = nn.Linear(config.hidden_size, NUM_PARTIES)
        self.sub = nn.Linear(config.hidden_size, NUM_SUBJECTS)

        self.loss_w = weights
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels, speaker_labels, party_labels, subject_labels):
        x = self.roberta(input_ids, attention_mask).last_hidden_state[:,0]
        x = self.dropout(x)

        lm, ls, lp, lsub = self.main(x), self.spk(x), self.party(x), self.sub(x)

        loss = (
            CrossEntropyLoss(self.loss_w)(lm, labels) +
            0.2 * CrossEntropyLoss()(ls, speaker_labels) +
            0.1 * CrossEntropyLoss()(lp, party_labels) +
            0.1 * CrossEntropyLoss()(lsub, subject_labels)
        )

        return {"loss": loss, "logits": lm}

model = RobertaMTL.from_pretrained(MODEL_NAME).to(device)

# ---------------- METRICS ----------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ---------------- TRAINING ----------------
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate(test_ds)

trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
