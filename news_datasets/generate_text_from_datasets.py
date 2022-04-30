


from datasets import load_dataset

# dataset = load_dataset('ccdv/cnn_dailymail')
raw_datasets = load_dataset(
            'cnn_dailymail', '3.0.0'
        )

raw_datasets['validation'].to_json('cnn-dm/val.json')
raw_datasets['test'].to_json('cnn-dm/test.json')

