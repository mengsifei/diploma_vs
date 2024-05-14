from tqdm import tqdm
import numpy as np
import torch
import gc
from train.evaluateChunk import evaluate_model_chunk

def train_model_chunk(model_doc, model_chunk, criteria, optimizers, schedulers, train_loader, val_loader, device, additional_info, epochs=10, early_stop=5, rubrics=['tr', 'cc']):
    best_val_loss = [np.inf] * len(rubrics)
    best_mae = [np.inf] * len(rubrics)
    best_kappa = [-np.inf] * len(rubrics)
    epochs_no_improve = 0
    n_epochs_stop = early_stop
    optimizers_doc, optimizers_chunk = optimizers
    schedulers_doc, schedulers_chunk = schedulers
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
            doc_inputs = {k: v.to(device) for k, v in batch[0].items() if k != 'labels'}
            chunk_inputs = {k: v.to(device) for k, v in batch[1].items() if k != 'labels'}
            labels = batch[0]['labels'].to(device)
            optimizers_doc.zero_grad()
            optimizers_chunk.zero_grad()
            doc_outputs = model_doc(doc_inputs['input_ids'], doc_inputs['attention_mask'], doc_inputs['token_type_ids'])
            chunk_outputs = model_chunk(chunk_inputs['input_ids'], chunk_inputs['attention_mask'], chunk_inputs['token_type_ids'], device)
            losses_doc = {rubrics[i]: criteria[0][i](doc_outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            losses_chunk = {rubrics[i]: criteria[1][i](chunk_outputs[:, i], labels[:, i]) * task_weights[i] for i in range(len(rubrics))}
            total_loss_doc = sum(losses_doc.values())
            total_loss_chunk = sum(losses_chunk.values())
            total_loss_doc.backward()
            total_loss_chunk.backward()
            optimizers_doc.step()
            optimizers_chunk.step()
            for rubric in rubrics:
                running_losses[rubric] += (losses_doc[rubric].item() + losses_chunk[rubric].item()) / 2 * labels.size(0)
            total_samples += labels.size(0)
        schedulers_doc.step()
        schedulers_chunk.step()
        avg_losses = {rubric: running_losses[rubric] / total_samples for rubric in rubrics}
        for rubric in rubrics:
            history[f'train_loss_{rubric}'].append(avg_losses[rubric])

        maes, kappas, valid_losses = evaluate_model_chunk((model_doc, model_chunk), val_loader, criteria, device, rubrics)
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
            checkpoint = { 
                'epoch': epoch,
                'model_chunk': model_chunk.state_dict(),
                'model_doc': model_doc.state_dict(),
                'optimizer_chunk': optimizers_chunk.state_dict(),
                'optimizer_doc': optimizers_doc.state_dict(),
                'scheduler_doc': schedulers_doc,
                'scheduler_chunk': schedulers_chunk}
            torch.save(checkpoint, f'checkpoints/best_model_{additional_info}.pth')
            print(f"Epoch {epoch+1}: New best model saved")
        epochs_no_improve = 0 if improved else epochs_no_improve + 1
        if epochs_no_improve >= n_epochs_stop:
            print(f'Epoch {epoch+1}: Early stopping triggered. No improvement for {n_epochs_stop} consecutive epochs.')
            break
        torch.cuda.empty_cache()
        gc.collect()
    return history
