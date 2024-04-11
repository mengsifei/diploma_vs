import matplotlib.pyplot as plt
import pandas as pd

def draw_graphs(results):
    subtitles = ['tr', 'cc', 'lr', 'gra']
    training_all = ["train_loss_{}".format(subtitle) for subtitle in subtitles]
    results[training_all] = results['train_loss'].apply(pd.Series)
    validation_all = ["validation_loss_{}".format(subtitle) for subtitle in subtitles]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(results[training_all], marker='.', label=training_all)
    axs[1].plot(results[validation_all], marker='.', label=validation_all)
    axs[0].set_title("Training loss comparison")
    axs[1].set_title("Validation loss comparison")
    for i in range(2):
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("MSE")
        axs[i].legend()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    kappas = ['kappa_scores_mean', 'kappa_tr', 'kappa_cc', 'kappa_lr', 'kappa_gra']
    maes = ['maes_mean', 'mae_tr', 'mae_cc', 'mae_lr', 'mae_gra']
    axs[0].plot(results[kappas], marker='.', label=kappas)
    axs[1].plot(results[maes], marker='.', label=maes)
    axs[0].set_title("QWK score comparison")
    axs[1].set_title("MAE comparison")
    axs[0].set_ylabel('QWK')
    axs[1].set_ylabel("MAE")
    for i in range(2):
        axs[i].set_xlabel("Epoch")
        axs[i].legend()
    plt.show()