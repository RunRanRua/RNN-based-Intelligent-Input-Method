import torch
from model import InputMethodModel
import config
from tokenizer import HFTokenizer

def predict_batch(model, input_tensor):
    """
    predict top 5 tokens for a batch of input tensors
    Parameters
    ----------
    model : torch.nn.Module
        the trained model
    input_tensor : torch.Tensor
        the input tensor of shape (batch_size, seq_len)
    Returns
    -------
    list of list of int
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)  # shape: (1, vocab_size)
    top5_idx = torch.topk(outputs, k=5).indices # shape: (batch_size, 5)
    top5_idx_list = top5_idx.tolist()
    return top5_idx_list


def predict(text, tokenizer, model, device):
    # 4. treat text
    idx = tokenizer.encode(text)
    input_tensor = torch.tensor([idx], dtype=torch.long).to(device)  # shape: (1, seq_len)

    # 5. predict
    top5_idx_list = predict_batch(model, input_tensor)  # shape: (1, 5)
    top5_tokens = [ tokenizer.idx2word[idx] for idx in top5_idx_list[0]]
    return top5_tokens


def run_predict():
    # 1. device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. get vocabulary
    tokenizer = tokenizer = HFTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    print("Vocabulary loaded.")

    # 3. load model
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'input_method_model.pth'))
    print("Model loaded.")


    input_history = ""

    print("Welcome to the Input Method Prediction System! (tape 'q' or 'quit' to quit)")
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit', 'Q']:
            print("See you next time!")
            break
        if user_input.strip() == "":
            print("Please enter some text.")
            continue
        input_history += user_input
        print("Current input history:", input_history)
        top5_predictions = predict(input_history, tokenizer, model, device)
        print("Top 5 predictions:", top5_predictions)
        input_history += " "
        

if __name__ == "__main__":
    run_predict()