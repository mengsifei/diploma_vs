from tqdm import tqdm
import numpy as np
import torch
import gc
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from train.evaluate import evaluate_model
def train_model_chunk(model_doc, model_chunk, criteria, optimizer, scheduler, train_loader, val_loader, device, additional_info, epochs=10, early_stop=5, rubrics=['tr', 'cc']):
    # Initialize best scores and stopping parameters
    best_val_loss = [np.inf] * len(rubrics)
    best_mae = [np.inf] * len(rubrics)
    best_kappa = [-np.inf] * len(rubrics)
    epochs_no_improve = 0
    n_epochs_stop = early_stop
    optimizer_doc, optimizer_chunk = optimizer
    scheduler_doc, scheduler_chunk = scheduler
    # Initialize history for each rubric and overall metrics
    history = {'kappa_scores_mean': [], 'maes_mean': []}
    for rubric in rubrics:
        history[f'train_loss_{rubric}'] = []
        history[f'validation_loss_{rubric}'] = []
        history[f'kappa_{rubric}'] = []
        history[f'mae_{rubric}'] = []

    task_weights = [1 / len(rubrics)] * len(rubrics)
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model_doc.train()
        model_chunk.train()
        running_losses = {rubric: 0.0 for rubric in rubrics}
        total_samples = 0

        for batch in train_loader:
            doc_inputs = batch[0].to(device)
            chunk_inputs = batch[1].to(device)
            labels = doc_inputs['labels']  # Assuming labels are shared and correctly formatted

            # Zero the parameter gradients
            optimizer_doc.zero_grad()
            optimizer_chunk.zero_grad()

            # Forward pass for both models
            doc_outputs = model_doc(doc_inputs['input_ids'], doc_inputs['attention_mask'], doc_inputs['token_type_ids'])
            chunk_outputs = model_chunk(chunk_inputs['input_ids'], chunk_inputs['attention_mask'], chunk_inputs['token_type_ids'])

            # Calculate losses for each rubric independently
            losses_doc = {rubrics[i]: criteria[i](doc_outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            losses_chunk = {rubrics[i]: criteria[i](chunk_outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}

            # Aggregate the losses from both models
            total_loss_doc = sum(losses_doc.values())
            total_loss_chunk = sum(losses_chunk.values())

            # Backward pass for both models
            total_loss_doc.backward()
            total_loss_chunk.backward()
            optimizer_doc.step()
            optimizer_chunk.step()

            # Record losses
            for rubric in rubrics:
                running_losses[rubric] += (losses_doc[rubric].item() + losses_chunk[rubric].item()) / 2 * labels.size(0)
            total_samples += labels.size(0)

        # Step the learning rate schedulers
        scheduler_doc.step()
        scheduler_chunk.step()

        # Log average losses and evaluate on validation data
        avg_losses = {rubric: running_losses[rubric] / total_samples for rubric in rubrics}
        for rubric in rubrics:
            history[f'train_loss_{rubric}'].append(avg_losses[rubric])

        maes, kappas, valid_losses = evaluate_model((model_doc, model_chunk), val_loader, criteria, device, rubrics)
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
            torch.save({'model_doc': model_doc.state_dict(), 'model_chunk': model_chunk.state_dict()}, f'checkpoints/best_model_{additional_info}.pth')
            print(f"Epoch {epoch+1}: New best model saved")

        epochs_no_improve = 0 if improved else epochs_no_improve + 1
        if epochs_no_improve >= n_epochs_stop:
            print(f'Epoch {epoch+1}: Early stopping triggered. No improvement for {n_epochs_stop} consecutive epochs.')
            break

        # Clear some memory
        torch.cuda.empty_cache()
        gc.collect()

    return history


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
        total_samples = 0

        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()

            outputs = model(**inputs)
            losses = {rubrics[i]: criteria[i](outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()

            for rubric in rubrics:
                running_loss[rubric] += losses[rubric].item() * labels.size(0)
            total_samples += labels.size(0)

        if scheduler:
            scheduler.step()

        # Calculate and log the average losses
        for rubric in rubrics:
            avg_train_loss = running_loss[rubric] / total_samples
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
