from utils.imports import *
from torch.utils.data import DataLoader
from torch.optim import Adam


def accuracy(predictions: FloatTensor, truth: LongTensor) -> int:
    correct_labels = torch.ones_like(predictions)
    correct_labels[predictions != truth] = 0

    return correct_labels.sum().item()


def measure_accuracy(model: Module, dl: DataLoader, device: str) -> float:
    correct = 0.
    total = 0.
    for X, Y in dl:
        preds = model(X.to(device))
        local_total, local_correct = accuracy(preds.argmax(dim=-1), Y.to(device))
        correct += local_correct
        total += local_total
    return correct / total


def train_batch(model: Module, X: FloatTensor, Y: LongTensor, loss_fn: tensor_map, opt: Adam) -> float:
    num_samples = X.shape[0] * X.shape[1]
    opt.zero_grad()
    pred_batch = model(X).view(num_samples, -1)
    batch_loss = loss_fn(pred_batch, Y.flatten())
    batch_loss.backward()
    opt.step()

    return batch_loss.item()


def train_epoch(model: Module, dl: DataLoader, loss_fn: tensor_map, opt: Adam, device: str) -> float:
    epoch_loss = 0.
    for batch_idx, (X, Y) in enumerate(dl):
        batch_loss = train_batch(model, X.to(device), Y.to(device), loss_fn, opt)
        epoch_loss += batch_loss
    epoch_loss /= batch_idx

    return epoch_loss


def eval_batch(model: Module, X: FloatTensor, Y: LongTensor, loss_fn: tensor_map, device: str) -> float:
    model.eval()
    num_samples = X.shape[0] * X.shape[1]
    pred_batch = model(X).view(num_samples, -1)

    predictions = pred_batch.max(1, keepdim=True)[1]
    batch_loss = loss_fn(pred_batch, Y.flatten())

    return batch_loss.item()


def eval_epoch(model: Module, dl: DataLoader, loss_fn: tensor_map, device: str) -> float:
    epoch_loss = 0.
    for batch_idx, (X, Y) in enumerate(dl):
        batch_loss = eval_batch(model, X, Y, loss_fn, device)
        epoch_loss += batch_loss
    epoch_loss /= batch_idx
    model.train()

    return epoch_loss


def train_model(model: Module, train_dl: DataLoader, val_dl: DataLoader, loss_fn: tensor_map,
                opt: Adam, num_epochs: int, device: str) -> List[Tuple[float, float]]:
    losses = []
    print('\nStarted training..')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dl, loss_fn, opt, device)
        val_loss = eval_epoch(model, val_dl, loss_fn, device)
        losses.append((train_loss, val_loss))
        accuracy = measure_accuracy(model, val_dl, device)
        print('Epoch {:d}/{:d}, training loss={:.4f}, validation loss={:.4f}, test accuracy={:.2f}%' \
              .format(epoch + 1, num_epochs, train_loss, val_loss, accuracy * 100))
    print('\nFinished training..')

    return losses

# plot training and validation losses
def plot_losses(losses: List[Tuple[float, float]]) -> None:
    train_losses = list(map(lambda l: l[0], losses))
    val_losses = list(map(lambda l: l[1], losses))
    num_epochs = len(train_losses)

    import matplotlib.pyplot as plt
    plt.title('Training progress over {} epochs'.format(num_epochs))
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy loss')
    plt.plot(list(range(1, num_epochs + 1)), train_losses, label='train')
    plt.plot(list(range(1, num_epochs + 1)), val_losses, label='eval')
    plt.legend()
    plt.show()
