import datasets

def create_dataset():
    data = {
        'tokens': [['HIV', 'is', 'a', 'virus'], ['Aspirin', 'is', 'a', 'medicine']],
        'ner_tags': [[1, 0, 0, 0], [1, 0, 0, 1]]
    }

    dataset = datasets.Dataset.from_dict(data)
    dataset = dataset.map(lambda examples: {'labels': examples['ner_tags']})
    return dataset
 