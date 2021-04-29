from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(labels, predictions, n_classes, keyword):
    """

    """

    cm = confusion_matrix(labels, predictions, labels=np.arange(0,n_classes))
    sum_per_label = np.sum(cm, axis=1)
    cm_norm = cm / sum_per_label[:, None]
    cm_norm =np.round(cm_norm, 2)
    fig = plt.figure(figsize=(15,7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f")
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.tight_layout()
    fig.savefig(f"confusionmatrix_{keyword}", dpi=150)


def plot_history(history, metrics_list, keyword):
    """

    """

    fig = plt.figure()
    for n, metric in enumerate(metrics_list):
        name = metric.replace("_", " ").capitalize()
        n_epochs = max(history.epoch)

        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(history.epoch, history.history['val_'+metric], label="Validation")
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.ylim([0,1])
        plt.legend(loc='upper left')
        plt.xticks(np.arange(0, n_epochs), np.arange(1, n_epochs))
    plt.tight_layout()
    fig.savefig(f"metrics_{keyword}", dpi=150)
