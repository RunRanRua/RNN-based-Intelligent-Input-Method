import torch
from dataset import get_dataloader
from model import InputMethodModel
import config
from predict import predict_batch
from tokenizer import HFTokenizer

def _evaluate(model, test_dataloader, device):
    top1_acc_correct = 0
    top5_acc_correct = 0
    total_cnt = 0
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)  # shape: (batch_size, seq_len)
        targets = targets.tolist() # shape: (batch_size,) e.g., [12, 45, 23, ...]

        top5_idx_list = predict_batch(model, inputs)  # shape: (batch_size, 5)  e.g., [[34,23,12,45,67], [23,45,12,78,90], ...]
        for target, top5_idx_list in zip(targets, top5_idx_list):
            total_cnt += 1
            if target == top5_idx_list[0]:
                top1_acc_correct += 1
            if target in top5_idx_list:
                top5_acc_correct += 1
    return top1_acc_correct / total_cnt, top5_acc_correct / total_cnt




def run_evaluate():
    # 1. device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. get vocabulary
    tokenizer = HFTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')
    print("Vocabulary loaded.")

    # 3. load model
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'input_method_model.pth'))
    print("Model loaded.")

    # 4. test data loader
    test_dataloader = get_dataloader(train=False, shuffle=False)

    # 5. evaluate
    top1_acc, top5_acc = _evaluate(model, test_dataloader, device)
    print("Evaluation Results:")
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")



if __name__ == "__main__":
    run_evaluate()