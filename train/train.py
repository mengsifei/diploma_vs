from tqdm import tqdm
import numpy as np
import torch
import gc
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from train.evaluate import *

def train_model(model, criteria, optimizer, scheduler, train_loader, val_loader, device, additional_info, epochs=10, early_stop=5, rubrics=['tr', 'cc']):
    best_val_loss = {rubric: np.inf for rubric in rubrics}
    best_mae = {rubric: np.inf for rubric in rubrics}
    best_kappa = {rubric: -np.inf for rubric in rubrics}
    epochs_no_improve = 0
    n_epochs_stop = early_stop
    history = {'train_loss': [], 'kappa_scores_mean': [], 'maes_mean': []}

    # Initialize history for each rubric
    for rubric in rubrics:
        history.update({
            f'validation_loss_{rubric}': [],
            f'kappa_{rubric}': [],
            f'mae_{rubric}': []
        })
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = sum(criteria[i](outputs[:, i], labels[:, i]) * 0.5 for i in range(len(rubrics)))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        if scheduler:
            scheduler.step()

        avg_train_loss = running_loss / total_samples
        history['train_loss'].append(avg_train_loss)

        # Evaluate the model
        maes, kappas, valid_losses = evaluate_model(model, val_loader, criteria, device, rubrics)

        # Update history with validation metrics
        for i, rubric in enumerate(rubrics):
            history[f'validation_loss_{rubric}'].append(valid_losses[i])
            history[f'kappa_{rubric}'].append(kappas[i])
            history[f'mae_{rubric}'].append(maes[i])

        # Calculate mean kappa and mean mae
        mean_kappa = np.mean(kappas)
        mean_mae = np.mean(maes)
        history['kappa_scores_mean'].append(mean_kappa)
        history['maes_mean'].append(mean_mae)

        print("Mean Validation QWK:", mean_kappa)
        print("Mean Validation MAE:", mean_mae)
        # Save model if there is an improvement in any of the rubrics
        improved = False
        for rubric in rubrics:
            if valid_losses[i] < best_val_loss[rubric]:
                best_val_loss[rubric] = valid_losses[i]
                improved = True
            if kappas[i] > best_kappa[rubric]:
                best_kappa[rubric] = kappas[i]
                improved = True
            if maes[i] < best_mae[rubric]:
                best_mae[rubric] = maes[i]
                improved = True

        if improved:
            torch.save(model.state_dict(), f'checkpoints/best_model_{additional_info}.pth')
            print(f"Epoch {epoch+1}: New best model saved")

        epochs_no_improve += 0 if improved else 1
        if epochs_no_improve >= n_epochs_stop:
            print(f'Epoch {epoch+1}: Early stopping triggered. No improvement for {n_epochs_stop} consecutive epochs.')
            break

        # Clear some memory
        torch.cuda.empty_cache()
        gc.collect()

    return history


def update_history(history, rubrics, maes, qwks, valid_loss, epoch, epochs):
    mae_mean = np.mean(maes)
    qwk_mean = np.mean(qwks)
    for i, rubric in enumerate(rubrics):
        history[f'validation_loss_{rubric}'].append(valid_loss[i])
        history[f'kappa_{rubric}'].append(qwks[i])
        history[f'mae_{rubric}'].append(maes[i])
    history['kappa_scores_mean'].append(qwk_mean)
    history['maes_mean'].append(mae_mean)
    print(f"Epoch {epoch+1}/{epochs}, Validation MAE: {mae_mean:.4f}, Validation QWK: {qwk_mean:.4f}")
    return history
