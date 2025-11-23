import config
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tokenizer import HFTokenizer


def _split_sent(text):
    # Split text into sentences based on punctuation.
    parts = re.split(r'([.!?])', text)
    sents = []

    for i in range(0, len(parts)-1, 2):
        sent = (parts[i] + parts[i+1]).strip()
        if sent:
            sents.append(sent)

    return sents

def _build_dataset(sentences,tokenizer):
    indexed_sentences = [tokenizer.encode(sentence) for sentence in sentences]
    
    dataset = []
    for sentence in tqdm(indexed_sentences, desc="Building dataset"):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def process():
    print("Processing data...")
    # 1. read data
    df = pd.read_csv(config.RAW_DATA_DIR / 'telecom.csv').sample(frac=0.05)
    # print(df.head())

    # 2. extract texts
    sentences =[]

    for dialog in df['text']:
        if isinstance(dialog, str):
            split_sents = _split_sent(dialog)
            sentences.extend(split_sents)

    # 3. split train and test sets
    train_sentences, test_setences = train_test_split(sentences, test_size=0.2)

    # 4. get vocabulary from training set
    HFTokenizer.build_vocab(train_sentences, config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 6. build train and test dataset
    tokenizer = HFTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    train_dataset = _build_dataset(train_sentences, tokenizer)
    test_dataset = _build_dataset(test_setences, tokenizer)

    # 7. save train and test sets
    pd.DataFrame(train_dataset).to_csv(config.PROCESSED_DATA_DIR / 'train.csv', index=False)
    pd.DataFrame(test_dataset).to_csv(config.PROCESSED_DATA_DIR / 'test.csv', index=False)
    
    print("Processed data saved.")

if __name__ == "__main__":
    process()