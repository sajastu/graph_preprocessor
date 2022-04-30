import os

from datasets import load_dataset

from path_vars import DS_DIR

os.makedirs('/dev/disk/data/cnn-dm/')


# dataset = load_dataset('ccdv/cnn_dailymail')
raw_datasets = load_dataset(
            'cnn_dailymail', '3.0.0'
        )

raw_datasets['train'].to_json(f'{DS_DIR}/train.json')
raw_datasets['validation'].to_json(f'{DS_DIR}/val.json')
raw_datasets['test'].to_json(f'{DS_DIR}/test.json')

