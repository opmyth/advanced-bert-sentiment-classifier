import random
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os
import logging

def train_model(model, train_loader, val_loader, config):
    wandb.init(project="text-classification", config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr = float(config['learning_rate']), weight_decay = config['weight_decay'])

    total_steps = len(train_loader) * config['num_epochs'] # number of batches * number of epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = config['warmup_steps'],
                                        num_training_steps = total_steps
                                    )

    best_val_accuracy = 0
    early_stopping_counter = 0

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0

        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc = f"Epoch {epoch + 1} / {config['num_epochs']}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['targets'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm']) #gradient clipping to prevent exploding gradients
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)

            train_preds.extend(preds.cpu().tolist()) #moving the predictions to cpu since the tolist() doesnt work on gpu
            train_labels.extend(labels.cpu().tolist())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_preds)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy
            })

        logging.info(f"Epoch {epoch + 1} / {config['num_epochs']}")
        logging.info(f"Average training loss: {avg_train_loss:.4f}")
        logging.info(f"Training_accuracy: {train_accuracy:.4f}")

        if (epoch + 1) % 2 == 0:
        
            val_accuracy, val_loss = evaluate_model(model, val_loader, criterion, device) 
            
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
                })

            logging.info(f"Validation Loss: {val_loss:.4f}")
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")


        #early sopping and model checkpoint

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), config['model_save_path'])
                logging.info(f"A new best model saved with validation accuracy: {best_val_accuracy:.4f}")
                early_stoppint_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config['early_stopping_threshold']:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    logging.info("Training completed :)")
    return model


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch["targets"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss
            _,preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy, avg_loss
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

def train_my_model(model, train_loader, val_loader, config):
    set_seed(config['seed'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    trained_model = train_model(model, train_loader, val_loader, config)
    return trained_model

