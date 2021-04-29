""" Dependencies """
# Load Huggingface transformer
from transformers import TFBertModel, BertConfig, BertTokenizerFast

# tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Libraries to import and process the data set
import pandas as pd
import numpy as np

# Metrics
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from tensorflow_addons.metrics import F1Score

# Utilities
from contextlib import redirect_stdout
import pathlib
import json

# Logging
#import wandb
#from wandb.keras import WandbCallback
#!wandb login


""" Directories and filenames """
# Define some names and directories
experiment_name = "1.1_BertLargeUncased" # name of the experiment
model_name = 'bert-large-uncased'
run_name = 'resampled'
base_dir = pathlib.Path(f"gdrive/MyDrive/Colab Notebooks/Thesis")
train_dataset = pathlib.Path.joinpath(base_dir, "train_resampled.csv")  # where to read the data from
test_dataset = pathlib.Path.joinpath(base_dir, "test_resampled.csv")

""" Parameters """
hyperparameters = {
    "max_token_length": 30,
    "batch_size": 32,
    "epochs": 1,
    "optimizer": {
        "name": "Adam",
        "learning_rate": 5e-05,
        "epsilon": 1e-08,
        "decay": 0.01,
        "clipnorm": 1.0
    },
    "random_state": 1
}


""" Data """
# Read data
train = pd.read_csv(train_dataset).sample(frac=1, random_state=hyperparameters["random_state"])
test = pd.read_csv(test_dataset).sample(frac=1, random_state=hyperparameters["random_state"])
n_classes = len(train["label"].unique()) # number of unique classes. since the train-test-split is stratified
                                                # we can be sure all classes are present in train

""" Wandb logging """
logging = True
if logging:
    run = wandb.init(project=experiment_name, sync_tensorboard=True)
    run.save()
    run.name = run_name
    save_dir = pathlib.Path(wandb.run.dir)
    wandb.config.update(hyperparameters)
else:
    save_dir = pathlib.Path.joinpath(base_dir, experiment_name)

""" Class weights """
# Determine class weights to reduce class imbalance
class_weight = {}
total_instances = len(train)
class_freqs = train["label"].value_counts()
for class_id, freq in zip(class_freqs.index, class_freqs):
    class_weight[class_id] = (1 / freq)*(total_instances)/n_classes
np.savetxt(pathlib.Path.joinpath(save_dir, "class_weight.txt"), list(class_weight.values()))
sample_weight = np.zeros(len(train))
for i, class_id in enumerate(train["label"]):
    sample_weight[i] = class_weight[class_id]

""" Build_tokenizer"""
# Load transformer configuration
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
config.save_pretrained(save_dir)

tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)

def tokenize(dataset):
    return tokenizer(
        dataset["abstract"].to_list(),
        truncation=True,
        padding=True,
        max_length=hyperparameters["max_token_length"],
        return_token_type_ids = False,
        return_attention_mask = False,
        return_tensors='tf'
    )

# tokenize train dataset
x_train = tokenize(train)
y_train = to_categorical(train["label"], num_classes=n_classes) # do one-hot encoding
# train_ds = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train, sample_weight))
# train_ds = train_ds.shuffle(buffer_size=len(train), seed=hyperparameters["random_state"]).batch(hyperparameters["batch_size"])

# tokenize test dataset
x_test = tokenize(test)
y_test = to_categorical(test['label'], num_classes=n_classes)

""" Build model """
# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config=config) # loads all pretrained weights
# Load the MainLayer
bert = transformer_model.layers[0]
# Build model input
inputs = Input(shape=hyperparameters["max_token_length"], name='input_ids', dtype='int64')
input_ids = {"input_ids": inputs}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(input_ids)[1]
# Add dropout layer for regularization
dropout_layer = Dropout(config.hidden_dropout_prob, name='regularization_layer') # dropout_prob=0.1
dropout = dropout_layer(bert_model, training=False)
# Add dense layer to condense to the number of classes
dense = Dense(units=n_classes,
              kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
              name='dense', activation='softmax')(dropout)
outputs = dense
# Combine it all in a model object
model = Model(inputs=input_ids, outputs=outputs, name=experiment_name)

# Save model summary
with open(pathlib.Path.joinpath(save_dir, 'model_summary.txt'), 'w') as file:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: file.write(x + '\n'))

# Set an optimizer
optimizer = Adam(
    learning_rate=hyperparameters["optimizer"]["learning_rate"],
    epsilon=hyperparameters["optimizer"]["epsilon"],
    decay=hyperparameters["optimizer"]["decay"],
    clipnorm=hyperparameters["optimizer"]["clipnorm"])

loss = CategoricalCrossentropy(from_logits=False)

metrics = [CategoricalAccuracy(name='accuracy'),
          Precision(name='precision'), # Precision is the percentage of predicted positives that were correctly classified
          Recall(name='recall'), # Recall is the percentage of actual positives that were correctly classified
          F1Score(num_classes=n_classes, average="macro",name="macro_f1",
)]

# Compile the model
model.compile(
  optimizer=optimizer,
  loss=loss,
  metrics=metrics)

""" Setup training """

# setup callbacks
checkpoint_callback = ModelCheckpoint(filepath=pathlib.Path.joinpath(save_dir, "checkpoint.ckpt"),
                              save_weights_only=True, verbose=1,
                              mode='auto', save_freq='epoch')
tensorboard_callback = TensorBoard(pathlib.Path.joinpath(save_dir, "logs"),
                                  histogram_freq=0, write_graph=False, write_images=False,
                                  update_freq=100)
if logging:
  wandb_callback = WandbCallback()
callbacks = [tensorboard_callback]


# train the model
history = model.fit(
    x=x_train["input_ids"],
    y=y_train,
    epochs=hyperparameters["epochs"],
    callbacks=callbacks,
    verbose=1,
    batch_size=hyperparameters["batch_size"]
    # class_weight=class_weight
    )

# Get predictions on test dataset
predictions = model.predict(
    x=x_test["input_ids"],
    batch_size=hyperparameters["batch_size"],
    verbose=1
)
prediction_max =np.argmax(predictions, axis=1) # select the prediction with highest probability
true_label = np.argmax(y_test, axis=1) # select the true label from one-hot encoding
pd.DataFrame([prediction_max, true_label], index=["prediction", "true_label"]).\
transpose().to_csv(pathlib.Path.joinpath(save_dir, "prediction-truth.csv"))

""" Metrics """

def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions, labels=np.arange(0,n_classes))
    sum_per_label = np.sum(cm, axis=1)
    cm_norm = cm / sum_per_label[:, None]
    cm_norm =np.round(cm_norm, 2)
    fig = plt.figure(figsize=(15,7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f")
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.tight_layout()
    fig.savefig(pathlib.Path.joinpath(save_dir, "confusion.png"), dpi=150)

plot_cm(true_label, prediction_max)

def plot_roc(labels, predictions):
    fig = plt.figure(figsize=(15, 13))
    linewidth = 2
    fpr = {}
    tpr = {}
    roc_auc = {}

    # encode
    labels_encoded = np.zeros([len(labels), n_classes])
    for i in range(len(labels_encoded)):
        labels_encoded[i, labels[i]] = 1

    # calculate
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_encoded[:,i], predictions[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.subplot(4, 4, i+1)
        plt.plot(fpr[i], tpr[i], linewidth=linewidth,
                 label='Label: %i (AUC = %0.2f)' % (i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='black', lw=linewidth, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1])
        plt.ylim([0, 1.05])
        plt.grid(True)
        plt.legend(loc='lower right')
        ax = plt.gca()
        ax.set_aspect('equal')
    fig.savefig(pathlib.Path.joinpath(save_dir, "roc.png"), dpi=150)

    # micro ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_encoded.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro ROC
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

    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"], linewidth=linewidth,
            label='macro: (AUC = %0.2f)' % (roc_auc["macro"]))
    plt.plot(fpr["micro"], tpr["micro"], linewidth=linewidth,
            label='micro: (AUC = %0.2f)' % (roc_auc["micro"]))
    plt.plot([0, 1], [0, 1], color='black', lw=linewidth, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(pathlib.Path.joinpath(save_dir, "roc_multiclass.png"), dpi=150)

plot_roc(true_label, predictions)

# Calculate metrics per label
cls_names =[str(cls) for cls in np.arange(0, n_classes)]
cls_report = classification_report(true_label, prediction_max, target_names=cls_names, output_dict=True)
cls_report = pd.DataFrame(cls_report).round(2).transpose()
cls_report.to_csv(pathlib.Path.joinpath(save_dir, f"metrics_report.csv"))

""" Submit results """
if logging:
    run.join()
    run.finish()