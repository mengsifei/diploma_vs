from tqdm import tqdm
import numpy as np
import torch
import gc
from train.evaluateDual import evaluate_model

def train_model(model, criteria, optimizer, scheduler, train_loader, val_loader, device, additional_info, epochs=10, early_stop=5, rubrics=['tr', 'cc']):
    # Initialize best scores and stopping parameters
    best_val_loss = [np.inf] * len(rubrics)
    best_mae = [np.inf] * len(rubrics)
    best_kappa = [-np.inf] * len(rubrics)
    epochs_no_improve = 0
    n_epochs_stop = early_stop

    # Initialize history for each rubric and overall metrics
    history = {'kappa_scores_mean': [], 'maes_mean': []}
    for rubric in rubrics:
        history[f'train_loss_{rubric}'] = []
        history[f'validation_loss_{rubric}'] = []
        history[f'kappa_{rubric}'] = []
        history[f'mae_{rubric}'] = []

    task_weights = [1 / len(rubrics)] * len(rubrics)
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = {rubric: 0.0 for rubric in rubrics}
        total_samples = {rubric: 0 for rubric in rubrics}
        for batch in train_loader:
            essay_inputs = {
                'essay_input_ids': batch['essay_input_ids'].to(device),
                'essay_attention_mask': batch['essay_attention_mask'].to(device),
                'essay_token_type_ids': batch['essay_token_type_ids'].to(device),
            }
            topic_inputs = {
                'topic_input_ids': batch['topic_input_ids'].to(device),
                'topic_attention_mask': batch['topic_attention_mask'].to(device),
                'topic_token_type_ids': batch['topic_token_type_ids'].to(device),
            }
            labels = batch['labels'].to(device)
            optimizer.zero_grad()

            outputs = model(essay_input_ids=essay_inputs['essay_input_ids'], 
                            essay_attention_mask=essay_inputs['essay_attention_mask'],
                            essay_token_type_ids=essay_inputs['essay_token_type_ids'], 
                            topic_input_ids=topic_inputs['topic_input_ids'], 
                            topic_attention_mask=topic_inputs['topic_attention_mask'],
                            topic_token_type_ids=topic_inputs['topic_token_type_ids'])
            
            losses = {rubrics[i]: criteria[i](outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            for i, rubric in enumerate(rubrics):
                running_loss[rubric] += losses[rubric].item() * labels.size(0)
                total_samples[rubric] += labels.size(0)
        if scheduler:
            scheduler.step()

        # Calculate and log the average losses
        for rubric in rubrics:
            avg_train_loss = running_loss[rubric] / total_samples[rubric]
            history[f'train_loss_{rubric}'].append(avg_train_loss)

        # Evaluate the model on validation data
        maes, kappas, valid_losses = evaluate_model(model, val_loader, criteria, device, rubrics)
        mean_kappa = np.mean(kappas)
        mean_mae = np.mean(maes)
        history['kappa_scores_mean'].append(mean_kappa)
        history['maes_mean'].append(mean_mae)

        print("Mean Validation QWK:", mean_kappa)
        print("Mean Validation MAE:", mean_mae)


        improved = False
        for i, rubric in enumerate(rubrics):
            history[f'validation_loss_{rubric}'].append(valid_losses[i])
            history[f'kappa_{rubric}'].append(kappas[i])
            history[f'mae_{rubric}'].append(maes[i])

        if np.mean(valid_losses) < np.mean(best_val_loss):
            best_val_loss = valid_losses
            improved = True
        if np.mean(kappas) > np.mean(best_kappa) and np.mean(maes) < np.mean(best_mae):
            best_kappa = kappas
            best_mae = maes
            improved = True

        if improved:
            torch.save(model.state_dict(), f'checkpoints/best_model_{additional_info}.pth')
            print(f"Epoch {epoch+1}: New best model saved")

        epochs_no_improve = 0 if improved else epochs_no_improve + 1
        if epochs_no_improve >= n_epochs_stop:
            print(f'Epoch {epoch+1}: Early stopping triggered. No improvement for {n_epochs_stop} consecutive epochs.')
            break

        # Clear some memory
        torch.cuda.empty_cache()
        gc.collect()

    return history
