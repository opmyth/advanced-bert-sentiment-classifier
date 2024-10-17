import yaml
import wandb
import torch
from transformers import BertTokenizer

from src.data.dataset import load_data, create_data_loaders
from src.model.classifier import BertClassifier
from src.train import train_my_model

def main():
    wandb.login()

    #loading them configurations
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    train_data, val_data, test_data = load_data(config['data_path'],
                                                test_size = config['test_size'],
                                                val_size = config['val_size'])


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, tokenizer, config['max_length'], config['batch_size']
        )

    model = BertClassifier(n_classes=config['n_classes'], dropout_rate=config['dropout_rate'])
    trained_model = train_my_model(model, train_loader, val_loader, config)

    torch.save(trained_model.state_dict(), config['final_model_path'])

    print("here we are done with the training letsgoo")
    
    wandb.finish()

if __name__ == "__main__":
    main()
