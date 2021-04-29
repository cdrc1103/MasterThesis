from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(labels, predictions):
    """

    """
    n_classes = labels.shape[1]
    fig = plt.figure(figsize=(15, 13))
    linewidth = 2
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:,i], predictions[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.subplot(4, 4, i+1)
        plt.plot(fpr[i], tpr[i], linewidth=linewidth,
                 label='Label: %i (AUC = %0.2f)' % (i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='darkorange', lw=linewidth, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1])
        plt.ylim([0, 1.05])
        plt.grid(True)
        plt.legend(loc='lower right')
        ax = plt.gca()
        ax.set_aspect('equal')
    plt.show()

    """Micro ROC"""
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    """Macro ROC"""
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    """Plot Micro + Macro"""
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"], linewidth=linewidth,
            label='macro: (AUC = %0.2f)' % (roc_auc["macro"]))
    plt.plot(fpr["micro"], tpr["micro"], linewidth=linewidth,
            label='micro: (AUC = %0.2f)' % (roc_auc["micro"]))
    plt.plot([0, 1], [0, 1], color='darkorange', lw=linewidth, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()