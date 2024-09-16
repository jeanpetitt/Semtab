from datasets import Dataset, DatasetDict
from huggingface_hub import login
import os
import pandas as pd

import re

def split_dataset(data, split_str):
    # Check if the split string starts with 'train[' and ends with '%]'
    if str(split_str).startswith("train"):
        match = re.match(r'train\[(\d+)%\]', split_str)
    elif str(split_str).startswith("val"):
        match = re.match(r'val\[(\d+)%\]', split_str)
    else:
        match = re.match(r'test\[(\d+)%\]', split_str)
    if not match:
        raise ValueError("Invalid format. Expected format: 'train[<number>%]'")
    
    # Extract the percentage as an integer
    percentage = int(match.group(1))
    
    # Calculate the size of the training data
    total_size = len(data)
    if str(split_str).startswith("train"):
        train_size = round(total_size * percentage / 100)
        # Split the data into train and remaining
        train_data = data[:train_size]
        remaining_data = data[train_size:]
        print(f"Training data size: {len(train_data)}")
        print(f"Remaining data size: {len(remaining_data)}")
        return train_data, remaining_data
    elif str(split_str).startswith("val"):
        val_size = round(total_size * percentage / 100)
        # Split the data into val and remaining
        val_data = data[:val_size]
        remaining_data = data[val_size:]
        print(f"validation data size: {len(val_data)}")
        print(f"Remaining data size: {len(remaining_data)}")
        return val_data, remaining_data
    else:
        test_size = round(total_size * percentage / 100)
        # Split the data into test and remaining
        test_data = data[:test_size]
        remaining_data = data[test_size:]
        print(f"validation data size: {len(test_data)}")
        print(f"Remaining data size: {len(remaining_data)}")
        return test_data, remaining_data



def open_csv(path, split=None):
    df = pd.read_csv(path)
    return df


def login_hub(token=None):
    if not token:
        print(token)
        return login(token=os.environ['HUB_TOKEN_Z'])
    return login(token=token)

def push_dataset_to_hub(
    train_path=None,
    repo_path=None,
    token=None,
    test_path=None, 
    val_path=None,
    dataset_path=None
    ):
    
    login_hub(token=token)
    train_dataset, test_dataset, val_dataset = '', '', ''
    dataset_dict = {}
    if not train_path and not test_path and not val_path and not dataset_path:
        raise ValueError("You should give at least the train path.")
    
    if not repo_path:
        raise ValueError("You should give a valid huggingFace repository. Example user/repository_name")
    
    if dataset_path:
        data = open_csv(dataset_path)
        train_data, train_remaining_data = split_dataset(data=data, split_str="train[80%]")
        train_dataset = Dataset.from_pandas(train_data)
        test_data, test_remaining_data = split_dataset(train_remaining_data, split_str="test[50%]")
        test_dataset = Dataset.from_pandas(test_data)
        val_data, test_remaining_data = split_dataset(test_remaining_data, split_str="val[100%]")
        val_dataset = Dataset.from_pandas(val_data)
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })
        dataset_dict.push_to_hub(repo_path)
        return dataset_dict
        
        
        
    
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
    
    