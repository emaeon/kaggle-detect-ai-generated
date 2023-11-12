import numpy as np
from datasets import load_dataset


# TO-DO: generated data oversampling by collecting external datasets

def make_dataset(dataset_path, valid_ratio, seed):
    
    dataset = load_dataset(
        'csv',
        data_files=dataset_path,
        )
    dataset = dataset.train_test_split(test_size=valid_ratio)
    
    return dataset['train'].shuffle(seed=seed), dataset['test']
