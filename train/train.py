import numpy as np
from tqdm import tqdm
import torch
from train.evaluate import evaluate_model
import gc

def train_model(model, criteria, optimizer, scheduler, train_loader, val_loader, device, additional_info, is_dual_version=False, epochs=10, early_stop=5):
    best_val_loss = [np.inf] * 4  # Initialize best validation loss
    best_mae = [np.inf] * 4
    best_qwk = [-np.inf] * 4
    epochs_no_improve = 0
    n_epochs_stop = early_stop
    rubrics = ['tr', 'cc', 'lr', 'gra']
    history = {'train_loss': [], 'kappa_scores_mean': [], 'maes_mean': []}
    past_losses = [[], [], [], []]  # List to store past losses for each task

    for rubric in rubrics:
        history.update({f'validation_loss_{rubric}': [], f'kappa_{rubric}': [], f'mae_{rubric}': []})
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        current_epoch_losses = [[], [], [], []]  # Temporary storage for current epoch losses
        running_losses = [0.0] * 4
        total_weights = [0.0] * 4
        task_weights = [1.0] * 4  # Default task weights
        
        # Update task weights based on past losses after the first epoch
        if epoch > 0:
            average_past_losses = [np.mean(losses) if losses else 1.0 for losses in past_losses]  # Avoid division by zero
            task_weights = [1.0 / (loss + 1e-6) for loss in average_past_losses]

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k.endswith('_ids') or k.endswith('_mask')}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            label_weights = batch['label_weights'].to(device).squeeze(1)
            losses = []
            for i in range(4):
                weighted_loss = criteria[i](outputs[:, i], labels[:, i])
                weighted_loss *= label_weights[:, i]
                final_loss = weighted_loss.mean() * 0.25 #* task_weights[i]  # Apply task-specific weight
                losses.append(final_loss)
                running_losses[i] += final_loss.item() * labels.size(0)  # Correct usage of .item()
                total_weights[i] += label_weights[:, i].sum().item()
                current_epoch_losses[i].append(final_loss.item())
            loss = sum(losses)
            loss.backward()
            optimizer.step()

        # Update past losses with the average of current epoch losses
        for i in range(4):
            past_losses[i].append(np.mean(current_epoch_losses[i]))

        if scheduler:
            scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

        avg_train_losses = [total_loss / total_weight if total_weight > 0 else 0 for total_loss, total_weight in zip(running_losses, total_weights)]
        history['train_loss'].append(avg_train_losses)
        print(f"============Average MSE Loss on Training=============\n {np.round(avg_train_losses, 4)}")

        maes, qwks, valid_loss = evaluate_model(model, val_loader, criteria, is_dual_version, device)
        history = update_history(history, rubrics, maes, qwks, valid_loss, epoch, epochs)
        improved = False
        if np.mean(valid_loss) < np.mean(best_val_loss) or ((np.mean(qwks) > np.mean(best_qwk))):
            torch.save(model.state_dict(), f'checkpoints/best_model_{additional_info}.pth')
            print(f"New best model saved at epoch {epoch+1}")
            improved = True
            best_val_loss = valid_loss if np.mean(valid_loss) < np.mean(best_val_loss) else best_val_loss
            best_mae = maes if np.mean(maes) < np.mean(best_mae) else best_mae
            best_qwk = qwks if np.mean(qwks) > np.mean(best_qwk) else best_qwk
        if improved:
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

