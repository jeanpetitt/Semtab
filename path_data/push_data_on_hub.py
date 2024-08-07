from datasets import Dataset, DatasetDict
from huggingface_hub import login
import os
import pandas as pd


def login_hub(token=None):
    if not token:
        return login(token=os.environ['HUB_TOKEN'])
    return login(token=token)

def push_dataset_to_hub(train_path=None, repo_path=None, token=None, test_path=None, val_path=None):
    
    login_hub(token=token)
    train_dataset, test_dataset, val_dataset = '', '', ''
    dataset_dict = {}
    if not train_path and not test_path and not val_path:
        raise ValueError("You should give at least the train path.")
    
    if not repo_path:
        raise ValueError("You should give a valid huggingFace repository. Example user/repository_name")
    
    if train_path:
        train_data = pd.read_csv(train_path, dtype=str)
        train_dataset = Dataset.from_pandas(train_data)
    
    if test_path:
        test_data = pd.read_csv(test_path, dtype=str)
        test_dataset = Dataset.from_pandas(test_data)
        
    if val_path:
        val_data = pd.read_csv(val_path, dtype=str)
        val_dataset = Dataset.from_pandas(val_data)
        
    if train_dataset and val_dataset and test_dataset:  
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })
    elif train_dataset and val_dataset:
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset
        }) 
    elif train_dataset and test_dataset:
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    elif val_dataset and test_dataset:
        dataset_dict = DatasetDict({
            "val": val_dataset,
            "test": test_dataset
        })
    else:
        dataset_dict = DatasetDict({
            "train": train_dataset
        })
    
    dataset_dict.push_to_hub(repo_path)
    return dataset_dict
    
    