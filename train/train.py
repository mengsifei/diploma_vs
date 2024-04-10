import numpy as np
from tqdm import tqdm
import torch
from train.evaluate import *

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, addtional_info, epochs=10):
    best_qwk = -np.inf  # Initialize best Quadratic Weighted Kappa
    history = {
        'train_loss': [],
        'validation_loss': [],
        'kappa_scores_mean': [],
        'maes_mean': [], 
        'kappa_tr': [],
        'kappa_cc': [],
        'kappa_lr': [],
        'kappa_gra': [],
        'mae_tr': [],
        'mae_cc': [],
        'mae_lr': [],
        'mae_gra': []
    }
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")
        # Evaluation
        maes, qwks, valid_loss = evaluate_model(model, val_loader, criterion, device)
        mae_mean = np.mean(maes)
        qwk_mean = np.mean(qwks)
        history['validation_loss'].append(valid_loss)
        history['kappa_scores_mean'].append(qwk_mean)
        history['maes_mean'].append(mae_mean)
        history['kappa_tr'].append(qwks[0])
        history['kappa_cc'].append(qwks[1])
        history['kappa_lr'].append(qwks[2])
        history['kappa_gra'].append(qwks[3])
        history['mae_tr'].append(maes[0])
        history['mae_cc'].append(maes[1])
        history['mae_lr'].append(maes[2])
        history['mae_gra'].append(maes[3])
        print(f"Validation - MAE Score: {mae_mean:.4f}, QWK: {qwk_mean:.4f}")
        if qwk_mean > best_qwk:
            best_qwk = qwk_mean
            torch.save(model.state_dict(), 'checkpoints/best_model_{}.pth'.format(addtional_info))
    return history