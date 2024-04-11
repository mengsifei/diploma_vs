import numpy as np
from tqdm import tqdm
import torch
from train.evaluate import *
import gc

def train_model(model, criteria, optimizer, scheduler, train_loader, val_loader, device, additional_info, epochs=10):
    best_val_loss = [np.inf, np.inf, np.inf, np.inf]  # Initialize best validation loss
    epochs_no_improve = 0  # Track epochs with no improvement
    n_epochs_stop = 6  # Number of epochs to stop after no improvement
    history = {
        'train_loss': [],
        'kappa_scores_mean': [],
        'maes_mean': []
    }
    for i, rubric in enumerate(rubrics):
        history['validation_loss_{}'.format(rubric)] = []
        history['kappa_{}'.format(rubric)] = []
        history['mae_{}'.format(rubric)] = []
    loss_weights = [0.25, 0.25, 0.25, 0.25]
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_losses = [0.0, 0.0, 0.0, 0.0]
        total_weights = [0.0, 0.0, 0.0, 0.0]  # Initialize total_weights for this epoch

        for batch in train_loader:
            # essay_inputs = {
            #     'essay_input_ids': batch['essay_input_ids'].to(device),
            #     'essay_attention_mask': batch['essay_attention_mask'].to(device)
            # }
            # topic_inputs = {
            #     'topic_input_ids': batch['topic_input_ids'].to(device),
            #     'topic_attention_mask': batch['topic_attention_mask'].to(device)
            # }
            # labels = batch['labels'].to(device)
            # label_weights = batch['label_weights'].to(device).squeeze(1)  # Squeeze the singleton dimension

            # optimizer.zero_grad()
            # outputs = model(essay_input_ids=essay_inputs['essay_input_ids'], 
            #                 essay_attention_mask=essay_inputs['essay_attention_mask'], 
            #                 topic_input_ids=topic_inputs['topic_input_ids'], 
            #                 topic_attention_mask=topic_inputs['topic_attention_mask'])

            # losses = []
            # for i in range(4):  # Assuming 4 subtasks
            #     weighted_loss = criteria[i](outputs[:, i], labels[:, i])
            #     weighted_loss *= label_weights[:, i]  # Apply weights element-wise
            #     final_loss = weighted_loss.mean() * loss_weights[i]  # Calculate mean of weighted losses
            #     losses.append(final_loss)
            #     running_losses[i] += final_loss.item() * label_weights[:, i].sum().item()  # Accumulate weighted losses
            #     total_weights[i] += label_weights[:, i].sum().item()  # Accumulate weights

            # loss = sum(losses)  # Combine the losses from all subtasks
            inputs = {k: batch[k].to(device) for k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
            losses = []
            for i in range(4):  # Assuming 4 subtasks
                loss = criteria[i](outputs[:, i], labels[:, i]) * loss_weights[i]
                losses.append(loss)
                running_losses[i] += loss.item()
            loss = sum(losses)  # Combine the losses
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            torch.cuda.empty_cache()
            gc.collect()

        avg_train_losses = [loss / len(train_loader) for loss in running_losses]
        history['train_loss'].append(avg_train_losses)
        # avg_train_losses = [total_loss / total_weight if total_weight > 0 else 0 for total_loss, total_weight in zip(running_losses, total_weights)]
        # history['train_loss'].append(avg_train_losses)
        print(f"============Average MSE Loss on Training=============\n {np.round(avg_train_losses, 4)}")

        # Evaluation on the validation set
        maes, qwks, valid_loss = evaluate_model(model, val_loader, criteria, device)
        mae_mean = np.mean(maes)
        qwk_mean = np.mean(qwks)
        rubrics = ['tr', 'cc', 'lr', 'gra']
        for i, rubric in enumerate(rubrics):
            history['validation_loss_{}'.format(rubric)].append(valid_loss[i])
            history['kappa_{}'.format(rubric)].append(qwks[i])
            history['mae_{}'.format(rubric)].append(maes[i])
        history['kappa_scores_mean'].append(qwk_mean)
        history['maes_mean'].append(mae_mean)
        print(f"Epoch {epoch+1}/{epochs}, Validation MAE: {mae_mean:.4f}, Validation QWK: {qwk_mean:.4f}")
        
        # Check for improvement and potentially save the model
        improved = False
        for i in range(4):  # Assuming 4 criteria
            if valid_loss[i] < best_val_loss[i]:
                best_val_loss[i] = valid_loss[i]
                improved = True

        if improved:
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model_{additional_info}.pth')
            print(f"New best model saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_epochs_stop:
            print(f'Early stopping triggered. No improvement in any criterion for {n_epochs_stop} consecutive epochs.')
            break
    return history
