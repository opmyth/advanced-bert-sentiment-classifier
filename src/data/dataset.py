import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class IMDBDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        self.reviews=reviews
        self.targets=targets
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review=str(self.reviews[index])
        target=self.targets[index]

        encoding=self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

        return {
            'review_text':review,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'targets':torch.tensor(target, dtype=torch.long)
        }

def load_data(data_path, test_size=0.1, val_size=0.1):
     df=pd.read_csv(data_path)
     reviews=df['review'].values
     targets=df['sentiment'].values
     targets=[1 if sample=='positive' else 0 for sample in targets]
        

     train_reviews,temp_reviews,train_targets,temp_targets=train_test_split(reviews,targets,test_size=(test_size+val_size),random_state=414)
     val_reviews,test_reviews,val_targets,test_targets=train_test_split(temp_reviews,temp_targets,test_size=test_size/(test_size+val_size),random_state=414)

     return (train_reviews,train_targets), (val_reviews,val_targets), (test_reviews,test_targets)

def create_data_loaders(train_data, val_data, test_data, tokenizer, max_length, batch_size):
    train_dataset=IMDBDataset(
            reviews=train_data[0],
            targets=train_data[1],
            tokenizer=tokenizer,
            max_length=max_length
        )

    
    val_dataset=IMDBDataset(
            reviews=val_data[0],
            targets=val_data[1],
            tokenizer=tokenizer,
            max_length=max_length
        )

    test_dataset=IMDBDataset(
            reviews=test_data[0],
            targets=test_data[1],
            tokenizer=tokenizer,
            max_length=max_length
        )

    train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

    val_loader=torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    test_loader=torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    return train_loader,val_loader,test_loader






























