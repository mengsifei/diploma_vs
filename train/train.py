import numpy as np
from tqdm import tqdm
import torch
from train.evaluate import evaluate_model
import gc

def train_model(model, criteria, optimizer, scheduler, train_loader, val_loader, device, additional_info, is_dual_version=False, epochs=10, early_stop=5):
    best_val_loss = [np.inf] * 4  # Initialize best validation loss for each task
    best_mae = [np.inf] * 4
    best_qwk = [-np.inf] * 4
    epochs_no_improve = 0
    n_epochs_stop = early_stop
    task_weights = [0.25] * 4
    rubrics = ['tr', 'cc', 'lr', 'gra']
    history = {'train_loss': [], 'kappa_scores_mean': [], 'maes_mean': []}

    # Initialize history for each rubric
    for rubric in rubrics:
        history.update({
            f'validation_loss_{rubric}': [],
            f'kappa_{rubric}': [],
            f'mae_{rubric}': []
        })
    
    total_samples = len(train_loader.dataset)  # Total number of samples

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_losses = [0.0] * 4  # Store sum of losses for each task
        task_samples_count = [0] * 4  # Count samples per task if varying batch sizes

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k.endswith('_ids') or k.endswith('_mask')}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()

            outputs = model(**inputs)
            losses = []

            for i in range(4):
                loss = criteria[i](outputs[:, i], labels[:, i])
                final_loss = loss * task_weights[i]  # Apply task-specific weights
                losses.append(final_loss)
                running_losses[i] += final_loss.sum().item()  # Sum up weighted losses
                task_samples_count[i] += labels.size(0)  # Assuming equal contribution from each sample

            loss = sum(losses)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

        avg_train_losses = [running_loss / task_sample_count for running_loss, task_sample_count in zip(running_losses, task_samples_count)]
        history['train_loss'].append(avg_train_losses)
        print(f"Average MSE Loss on Training: {np.round(avg_train_losses, 4)}")

        maes, qwks, valid_loss = evaluate_model(model, val_loader, criteria, is_dual_version, device)
        history = update_history(history, rubrics, maes, qwks, valid_loss, epoch, epochs)
        
        improved = False
        for i in range(4):
            if valid_loss[i] < best_val_loss[i] or (qwks[i] > best_qwk[i] and maes[i] < best_mae[i]):
                improved = True
                best_val_loss[i] = valid_loss[i]
                best_mae[i] = maes[i]
                best_qwk[i] = qwks[i]

        if improved:
            torch.save(model.state_dict(), f'checkpoints/best_model_{additional_info}.pth')
            print(f"New best model saved at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= n_epochs_stop:
            print(f'Early stopping triggered. No improvement for {n_epochs_stop} consecutive epochs.')
            break

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
