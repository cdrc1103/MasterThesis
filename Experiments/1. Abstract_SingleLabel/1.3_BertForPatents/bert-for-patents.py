""" Dependencies """
# Load Huggingface transformer
from transformers import BertModel, BertConfig, BertTokenizerFast, TFPreTrainedModel, PreTrainedModel

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Some addons
from tensorflow_addons.metrics import F1Score

# Libraries to import and process the data set
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# Utilities
from contextlib import redirect_stdout
import pathlib
import json

""" Directories and filenames """
# Define some names
experiment_name = "1.3_BertForPatents" # name of the experiment
version = "v3"
# base_dir = pathlib.Path(f"gdrive/MyDrive/Colab Notebooks/Thesis/{experiment_name}")
base_dir = pathlib.Path("")
train_dataset = pathlib.Path.joinpath(base_dir, f"train_{version}.csv")  # where to read the data from
test_dataset = pathlib.Path.joinpath(base_dir, f"test_{version}.csv")
model_name = 'bert-for-patents'
# Read data
train = pd.read_csv(train_dataset)
test = pd.read_csv(test_dataset)
n_classes = len(train["label"].unique()) # number of unique classes. since the train-test-split is stratified
                                                # we can be sure all classes are present in train

""" Class weights """
# Determine class weights to tackle class imbalance
class_weight = {}
total_instances = len(train)
class_freqs = train["label"].value_counts()
for class_id, freq in zip(class_freqs.index, class_freqs):
    class_weight[class_id] = (1 / freq)*(total_instances)/2.0
class_weight = pd.DataFrame(class_weight, index=class_weight.keys())
class_weight.to_csv(pathlib.Path.joinpath(base_dir, f"class_weight_{version}.csv"))
sample_weight = []
for class_id in train["label"]:
    sample_weight.append(class_weight[class_id])


""" Parameters """
# Dataset
max_token_length = 30 # number of tokens per example

# Training
batch_size = 32
epochs = 1 # iterate over full dataset x times
random_state = 1
prefetch_size = 2

# Adam Optimizer
learning_rate = 5e-05
epsilon = 1e-08
decay = 0.01
clipnorm = 1.0

hyperparameters = {
    "max_token_length": max_token_length,
    "batch_size": batch_size,
    "epochs": epochs,
    "optimizer": {
        "name": "Adam",
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "decay": decay,
        "clipnorm": clipnorm
    }
}
with open(pathlib.Path.joinpath(base_dir, f"hyperparameters_{version}.json"), 'w') as file:
    json.dump(hyperparameters, file)

""" Callbacks """
checkpoint_callback = ModelCheckpoint(filepath=pathlib.Path.joinpath(base_dir, f"checkpoint_{version}.ckpt"),
                              save_weights_only=True, verbose=1,
                              mode='auto', save_freq='epoch')
tensorboard_callback = TensorBoard(pathlib.Path.joinpath(base_dir, f"logs_{version}"),
                                   histogram_freq=1, write_graph=False, write_images=False,
                                   update_freq=100)

""" Transformer and tokenizer config """
# Load transformers config
config = BertConfig.from_json_file("bert_for_patents_large_config.json")
config.output_hidden_states = False
with open(pathlib.Path.joinpath(base_dir, f'config_{version}.txt'), 'w') as file: # save config to dir
    with redirect_stdout(file):
        print(config)

tokenizer = BertTokenizerFast("bert_for_patents_vocab_39k.txt")

def tokenize(dataset):
    return tokenizer(
        dataset["abstract"].to_list(),
        truncation=True,
        padding=True,
        max_length=max_token_length,
        return_token_type_ids = False,
        return_attention_mask = False,
        return_tensors='tf'
  )

# Train
x_train = tokenize(train)
y_train = to_categorical(train["label"], num_classes=n_classes)
train_ds = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))
train_ds = train_ds.shuffle(buffer_size=len(train), seed=random_state).batch(batch_size)

# Test
x_test = tokenize(test)
y_test = to_categorical(test['label'], num_classes=n_classes)

""" Model architecture """
# Load the Transformers BERT model
transformer_model = BertModel.from_pretrained("/home/cedric/Documents/Data/tf_model/model.ckpt.index", config=config, from_tf=True)
transformer_model.save_pretrained("/home/cedric/Documents/MasterThesis/Experiments/1. Abstract_SingleLabel/1.3_BertForPatents")
transformer_model = TFPreTrainedModel.from_pretrained(transformer_model, config=config, from_pt=True) # loads all pretrained weights
# Load the MainLayer
bert = transformer_model.layers[0]

# Build model input
inputs = Input(shape=max_token_length, name='input_ids', dtype='int64')
input_ids = {"input_ids": inputs}

# Build model output
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

# Print model summary and save it
with open(pathlib.Path.joinpath(base_dir, f'model_summary_{version}.txt'), 'w') as file:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: file.write(x + '\n'))

# Set an optimizer
optimizer = Adam(
    learning_rate=learning_rate,
    epsilon=epsilon,
    decay=decay,
    clipnorm=clipnorm)

# Set loss and metrics
loss = CategoricalCrossentropy(from_logits=False)

metric = [CategoricalAccuracy(name='accuracy'),
          Precision(name='precision'), # Precision is the percentage of predicted positives that were correctly classified
          Recall(name='recall'), # Recall is the percentage of actual positives that were correctly classified
          F1Score(name='macro_f1', num_classes=n_classes, average='macro')]

""" Training """
# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metric)

# Fit the model
history = model.fit(
    x=train_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback],
    verbose=1
    )

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(pathlib.Path.joinpath(base_dir, f"history_{version}.csv"))

""" Evaluation """

# Evaluate
results = model.evaluate(x_test["input_ids"], y_test, batch_size=batch_size)
results_df = pd.DataFrame(results, index=["loss", "accuracy", "precision", "recall", "macro_f1"])
results_df.to_csv(pathlib.Path.joinpath(base_dir, f"tf_evaluation{version}.csv"))
print(results_df)

# Predictions on test dataset
predictions = model.predict(
    x=x_test["input_ids"],
    batch_size=batch_size,
    verbose=1
)
np.savetxt(f'predictions{version}.csv', predictions, delimiter=',')
prediction_max =np.argmax(predictions, axis=1) # select the prediction with highest probability
true_label = np.argmax(y_test, axis=1) # select the true label from one-hot encoding

# Create confusion matrix from results
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
    fig.savefig(pathlib.Path.joinpath(base_dir, f"confusion_{version}"), dpi=150)

plot_cm(true_label, prediction_max)

# Calculate metrics per label
cls_names =[str(cls) for cls in np.arange(0, n_classes)]
cls_report = classification_report(true_label, prediction_max, target_names=cls_names, output_dict=True)
cls_report = pd.DataFrame(cls_report).round(2)
cls_report.to_csv(pathlib.Path.joinpath(base_dir, f"metrics_report_{version}.csv"))