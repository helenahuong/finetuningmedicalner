import numpy as np
from datasets import load_metric

ner_labels = {
    '0': 'O',     # Outside of a named entity
    '1': 'B-DIS', # Beginning of a disease
    '2': 'I-DIS', # Inside of a disease
    '3': 'B-MED', # Beginning of a medication
    '4': 'I-MED'  # Inside of a medication
}

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[ner_labels[str(l)] for l in label if l != -100] for label in labels]
    true_predictions = [[ner_labels[str(p)] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
