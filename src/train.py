import torch
from dataset import get_dataloader
from model import InputMethodModel
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from tokenizer import HFTokenizer

def _train_one_epoch(model, dataloader, loss_func, optimizer, device):
    """
    train one epoch
    Parameters
    ----------
    model : torch.nn.Module
        the model to train
    dataloader : iterable
        the training data
    loss_func : torch.nn.Module
        the loss function
    optimizer : torch.optim.Optimizer
        the optimizer
    device : torch.device
        the device to use
    Returns
    -------
    float
        the average of training loss for this epoch
    """
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device)  # shape: (batch_size, seq_len)
        targets = targets.to(device) # shape: (batch_size,)

        # forward
        outputs = model(inputs) # shape: (batch_size, vocab_size)
        loss = loss_func(outputs, targets)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def train():
    # 1. device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. get data loader
    dataloader = get_dataloader()

    # 3. get vocabulary
    tokenizer = HFTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'vocab.txt')

    # 4. model
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)

    # 5. loss func
    loss__func = torch.nn.CrossEntropyLoss()

    # 6. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 7. tensorboard writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 8. training loop
    best_loss = float('inf')
    for epoch in range(1, 1 + config.EPOCHS):
        print("=" * 10, f"Epoch: {epoch}/{config.EPOCHS}", "=" * 10)
        # training 1 epoch
        loss = _train_one_epoch(model, dataloader, loss__func, optimizer, device)
        print(f"loss: {loss}")

        # record the result
        writer.add_scalar('Loss', loss, epoch)
    
    writer.close()

    # 9. save the model
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), config.MODELS_DIR / 'input_method_model.pth')
        print("Model saved.")





if __name__ == "__main__":
    train()