import sys
import os

# Add the parent directory to the sys.path to resolve the import issue
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data.create_dataset import create_dataset
from utils.preprocess import tokenize_and_align_labels, data_collator

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

def train_model():
    dataset = create_dataset()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=5)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator(tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./results")

if __name__ == "__main__":
    train_model()
