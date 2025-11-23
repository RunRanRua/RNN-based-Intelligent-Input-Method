from transformers import AutoTokenizer
from tqdm import tqdm
import config

class HFTokenizer:
    unk_token = '<unk>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_token_id = self.word2idx[self.unk_token]        
 
    @staticmethod
    def tokenize(text):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer.tokenize(text)
    
    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.unk_token_id) for token in tokens]    
    
    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc="Building vocabulary"):
            tokens = cls.tokenize(sentence)
            vocab_set.update(tokens)
        
        vocab_list = [cls.unk_token] + list(vocab_set)

        # 5. save vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))
    
    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)
    
if __name__ == "__main__":
    tokenizer = HFTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    print(f"vocabulary size: {tokenizer.vocab_size}")
    print(f"unknown token: {tokenizer.unk_token}")
    print(tokenizer.encode("Hello, how are you?"))
