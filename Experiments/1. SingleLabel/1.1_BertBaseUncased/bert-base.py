""" Dependencies """
# Load Huggingface transformer
from transformers import TFBertModel, BertConfig, BertTokenizerFast, TFBertForSequenceClassification

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.utils import to_categorical

# Some addons
from tensorflow_addons.metrics import F1Score

# Libraries to import and process the data set
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pprint import pprint

# Utilities
from contextlib import redirect_stdout
import pathlib
import json

""" Directories and filenames """
# Define some names
experiment_name = "1.1_BertBaseUncased" # name of the experiment
version = "v3"
base_dir = pathlib.Path(f"gdrive/MyDrive/Colab Notebooks/Thesis/{experiment_name}")
train_dataset = pathlib.Path.joinpath(base_dir, "train.csv")  # where to read the data from
test_dataset = pathlib.Path.joinpath(base_dir, "test.csv")
model_name = 'bert-base-uncased'
# Read data
train = pd.read_csv(train_dataset)
train = train.sample(frac=1, random_state=1) # shuffle dataset
test = pd.read_csv(test_dataset)
test = test.sample(frac=1, random_state=1) # shuffle dataset
n_classes = len(train["label_encoded"].unique()) # number of unique classes. since the train-test-split is stratified
                                                # we can be sure all classes are present in train

""" Class weights """
# Determine class weights to tackle class imbalance
class_weight = {}
total_instances = len(train)
class_freqs = train["label_encoded"].value_counts()
for class_id, freq in zip(class_freqs.index, class_freqs):
    class_weight[class_id] = (1 / freq)*(total_instances)/2.0
sample_weight = []
for class_id in train["label_encoded"]:
    sample_weight.append(class_weight[class_id])

""" Parameters """
# Dataset
max_token_length = 200 # number of tokens per example
validation_split = 0.2 # is splitted from the training set

# Training
train_batch_size = 16
validation_batch_size = 16
test_batch_size = 16
steps_per_epoch = 1000 # useful because metrics are logged more often
epochs = 21 # iterate over full dataset x times

# Adam Optimizer
learning_rate = 5e-05
epsilon = 1e-08
decay = 0.01
clipnorm = 1.0

hyperparameters = {
    "max_token_length": max_token_length,
    "validation_split": validation_split,
    "train_batch_sizer": train_batch_size,
    "validation_batch_size": validation_batch_size,
    "test_batch_size": test_batch_size,
    "steps_per_epoch": steps_per_epoch,
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
# Callbacks
cp_callback = ModelCheckpoint(filepath=pathlib.Path.joinpath(base_dir, f"checkpoint_{version}.ckpt"),
                              save_weights_only=True, verbose=1)
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2,     # Reduce learning rate if val_loss meats a plateau
                                patience=5, min_lr=0.001)

""" Transformer and tokenizer config """
# Load transformers config and set output_hidden_states to False ---> Why?
# from_pretrained loads weights from pretrained model
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# print(f"Model config: {config}")
with open(pathlib.Path.joinpath(base_dir, f'config_{version}.txt'), 'w') as file: # save config to dir
    with redirect_stdout(file):
        print(config)

tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=model_name, config=config)
x_train = tokenizer(
    text=train['abstract'].to_list(),
    add_special_tokens=True,
    max_length=max_token_length,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=False,
    verbose=True)
y_train = to_categorical(train["label_encoded"], num_classes=n_classes)
x_test = tokenizer(
    text=test['abstract'].to_list(),
    add_special_tokens=True,
    max_length=max_token_length,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=False,
    verbose=True)
y_test = to_categorical(test["label_encoded"], num_classes=n_classes)

""" Model architecture """
# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config=config) # loads all pretrained weights
# Load the MainLayer
bert = transformer_model.layers[0]

# Build model input
input_ids = Input(shape=max_token_length, name='input_ids', dtype='int64')
inputs = {'input_ids': input_ids}

# Build model output
# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
# Add dropout layer for regularization
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
# Add dense layer to condense to the number of classes
dense = Dense(units=n_classes,
              kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
              name='dense', activation='softmax')(
        pooled_output)
outputs = {'label': dense}
# Combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name=experiment_name)
# model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

# Print model structure and save it
with open(pathlib.Path.joinpath(base_dir, f'summary_{version}.txt'), 'w') as file:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: file.write(x + '\n'))

# Set an optimizer
optimizer = Adam(
    learning_rate=learning_rate,
    epsilon=epsilon,
    decay=decay,
    clipnorm=clipnorm)

# Set loss and metrics
loss = {'label': CategoricalCrossentropy(from_logits=True)}

metric = [CategoricalAccuracy(name='accuracy'),
          Precision(name='precision'), # Precision is the percentage of predicted positives that were correctly classified
          Recall(name='recall'), # Recall is the percentage of actual positives that were correctly classified
          F1Score(name='f1score', num_classes=n_classes, average='macro')]

""" Training """
# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metric)

# Fit the model
history = model.fit(
    x={'input_ids': np.array(x_train["input_ids"])},
    y={'label': np.array(y_train)},
    validation_split=validation_split,
    batch_size=train_batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_batch_size=validation_batch_size,
    #callbacks=[cp_callback], #!!!lr_callback],
    verbose=1,
    sample_weight=np.array(sample_weight))

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(pathlib.Path.joinpath(base_dir, f"history_{version}.csv"))

"""" Evaluation """
# plot the metric history from training
def plot_metrics(history):
    metrics = ['accuracy', 'f1score', 'precision', 'recall']
    fig = plt.figure()
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(history.epoch, history.history['val_'+metric], label="Validation")
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.ylim([0,1])
        plt.legend(loc='upper left')
        plt.xticks(np.arange(0, epochs), np.arange(1, epochs))
    plt.tight_layout()
    fig.savefig(pathlib.Path.joinpath(base_dir, f"metrics_{version}"), dpi=150)

plot_metrics(history)

# # Evaluate on test data
# result = model.evaluate(
#     x={'input_ids': np.array(x_test['input_ids'])},
#     y={'label': np.array(y_test)},
#     batch_size=test_batch_size
# )
# model_eval = dict(zip(model.metrics_names, result))
# pprint(f"Evaluation metrics:\n{model_eval}")
# pd.DataFrame(model_eval).to_csv(f"result_{version}.csv")


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

predictions = model.predict(
    x={'input_ids': np.array(x_test['input_ids'])},
    batch_size=test_batch_size
)
# Create confusion matrix from results
predictions =np.argmax(predictions["label"], axis=1) # select the prediction with highest probability
y_test = np.argmax(y_test, axis=1) # select the true label from one-hot encoding
plot_cm(y_test, predictions)

# Calculate classification report
cls_names =[str(cls) for cls in np.arange(0, n_classes)]
cls_report = classification_report(y_test, predictions, target_names=cls_names, output_dict=True)
cls_report = pd.DataFrame(cls_report).round(2)
cls_report.to_csv(pathlib.Path.joinpath(base_dir, f"metrics_report_{version}.csv"))