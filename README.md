# BERT-based Text Classification for IMDB Reviews

This project implements a BERT-based text classification model to perform sentiment analysis on IMDB movie reviews. It uses PyTorch and the Hugging Face Transformers library to fine-tune a BERT model for binary classification (positive/negative sentiment).

## Project Structure

```
text-classification/
├── data/
│   └── imdb_data.csv
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── model/
│   │   └── classifier.py
│   ├── train.py
├── main.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-classification.git
   cd text-classification
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create --name bert-classifier

   conda activate bert-classifier
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place your IMDB dataset (CSV format) in the `data/` directory.
   - Ensure your CSV file has 'review' and 'sentiment' columns.

5. Configure the project:
   - Edit `config.yaml` to set your desired parameters, data paths, and model settings.

## Usage

To train the model:
   ```bash
   python main.py
   ```

## Features

- BERT-based text classification
- Custom attention pooling mechanism
- Training with early stopping
- Weights & Biases integration for experiment tracking (optional)

## Configuration

The `config.yaml` file contains all the configurable parameters for the project. Key configurations include:

- Data paths
- Model hyperparameters (e.g., learning rate, batch size)
- Training settings (e.g., number of epochs, early stopping threshold)

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## Acknowledgments

- The BERT model implementation is based on the Hugging Face Transformers library.
- The IMDB dataset used for this project is publicly available.
