import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from utils.model.simple_classifier import SimpleClassifier
from utils.data.load_image_data import load_image_data
import numpy as np
from utils.common.utils import seed_fix

def train_classifier(data_dir, epochs=5, batch_size=8, lr=1e-3, save_path='classifier.pth'):
    seed = 430
    seed_fix(seed)
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    train_loader = load_image_data(train_dir, batch_size=batch_size, shuffle=True)
    val_loader = load_image_data(val_dir, batch_size=batch_size, shuffle=True)
    model = SimpleClassifier().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    log_path = Path(__file__).parent.parent / 'result' / 'classifier' / 'train_log.npy'
    log_arr = np.zeros((epochs, 5), dtype=np.float32)  # epoch, train_loss, train_acc, val_loss, val_acc
    for epoch in range(epochs):
        # Train
        model.train()
        total, correct = 0, 0
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
                
            outputs = model(images)
            if i == 0:
                print(f"[DEBUG][Train] images: {images.shape}, labels: {labels.shape}, labels unique: {labels.unique()}, dtype: {labels.dtype}")
                print(f"[DEBUG][Train] outputs: {outputs.shape}, sample: {outputs[0].detach().cpu().numpy()}")
                preds_dbg = outputs.argmax(1)
                print(f"[DEBUG][Train] preds: {preds_dbg.cpu().numpy()}, labels: {labels.cpu().numpy()}")
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_total, val_correct = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                images, labels = batch
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                if j == 0:
                    print(f"[DEBUG][Val] images: {images.shape}, labels: {labels.shape}, labels unique: {labels.unique()}, dtype: {labels.dtype}")
                    print(f"[DEBUG][Val] outputs: {outputs.shape}, sample: {outputs[0].detach().cpu().numpy()}")
                    preds_dbg = outputs.argmax(1)
                    print(f"[DEBUG][Val] preds: {preds_dbg.cpu().numpy()}, labels: {labels.cpu().numpy()}")
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_running_loss += loss.item() * labels.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        log_arr[epoch] = [epoch+1, train_loss, train_acc, val_loss, val_acc]
    np.save(log_path, log_arr)
    print(f"Train/Val log saved to {log_path}")

    # Save model in unified format (dict) to result/classifier/model.pt
    result_dir = Path(__file__).parent.parent / 'result' / 'classifier'
    result_dir.mkdir(parents=True, exist_ok=True)
    save_path = result_dir / 'model.pt'
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': None,
        'best_val_loss': None,
        'exp_dir': str(result_dir)
    }, save_path)
    print(f"Model saved to {save_path}")


    # MRISliceDataset is now provided by utils.data.load_image_data, so local definition is removed.

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    args = parser.parse_args()
    # Use user-provided seed if given
    from utils.common.utils import seed_fix
    seed_fix(args.seed)
    train_classifier(args.data_dir, args.epochs, args.batch_size, args.lr)
