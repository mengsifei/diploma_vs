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
        running_loss = {rubric: 0.0 for rubric in rubrics}
        total_samples = 0

        for batch in train_loader:
            doc_input, seg_input = batch  # Unpacking document and segment inputs
            labels = doc_input['labels'].to(device)  # Assumes labels are the same for both inputs
            optimizer.zero_grad()
            
            # Processing document-level inputs
            doc_predictions = model_doc(doc_input['input_ids'].to(device), 
                                        doc_input['attention_mask'].to(device), 
                                        doc_input['token_type_ids'].to(device))
            doc_predictions = torch.squeeze(doc_predictions)

            # Processing chunk/segment-level inputs
            chunk_predictions = []
            for seg_idx in range(len(seg_input['input_ids'])):
                seg_pred = model_chunk(seg_input['input_ids'][seg_idx].to(device), 
                                       seg_input['attention_mask'][seg_idx].to(device), 
                                       seg_input['token_type_ids'][seg_idx].to(device))
                chunk_predictions.append(torch.squeeze(seg_pred))

            # Combine predictions from all chunks
            batch_predictions = torch.mean(torch.stack(chunk_predictions), dim=0)

            # Combine document and chunk level predictions
            batch_predictions = (doc_predictions + batch_predictions) / 2

            # Compute Loss
            loss = criteria(batch_predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * doc_input['input_ids'].size(0)
            total_samples += doc_input['input_ids'].size(0)

        if scheduler:
            scheduler.step()

        # Calculate and log the average losses
        for rubric in rubrics:
            avg_train_loss = running_loss[rubric] / total_samples
            history[f'train_loss_{rubric}'].append(avg_train_loss)

        # Evaluate the model on validation data
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
