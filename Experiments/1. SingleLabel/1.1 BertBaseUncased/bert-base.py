# Load Huggingface transformer
from transformers import TFBertModel, BertConfig, BertTokenizerFast

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# Libraries to import and process the data set
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt

# Utilities
from contextlib import redirect_stdout

""" Parameters """

# Names
experiment_name = "1.1 BertBaseUncased" # name of the experiment
dataset_file = "dataset.csv"  # where to read the data from
model_name = 'bert-base-uncased'

# Early stopping
acc_threshold = 0.99

# Dataset
max_token_length = 100 # number of tokens per example
test_size = 0.5
validation_split = 0.1 # is splitted from the training set

# Training
batch_size = 16
epochs = 3

# Adam Optimizer
learning_rate = 5e-05
epsilon = 1e-08
decay = 0.01
clipnorm = 1.0

""" ----------- """

# Callbacks
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > acc_threshold:
            print(f"\nReached {acc_threshold}% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()


# Read data
dataset = pd.read_csv(dataset_file)
n_classes = len(dataset["label_encoded"].unique())

# Load transformers config and set output_hidden_states to False ---> Why?
# from_pretrained loads weights from pretrained model
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# print(f"Model config: {config}")
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
with open(f'config_{timestamp}.txt', 'w') as file: # save config to dir
    with redirect_stdout(file):
        print(config)

# Split data and tokenize the input
train, test = train_test_split(dataset, test_size=test_size, random_state=10000, shuffle=True) # random state for reproducability
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

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config=config) # loads all pretrained weights
# Load the MainLayer
bert = transformer_model.layers[0]

# Build model input
input_ids = Input(shape=max_token_length, name='input_ids', dtype='int64')
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Build model output
label = Dense(units=n_classes,
              kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
              name='dense')(
        pooled_output)
outputs = {'label': label}

# Combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name=experiment_name)

# Print model structure and save it
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
with open(f'summary_{timestamp}.txt', 'w') as file:
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
# Using from_logits=true means that the prediction tensor is one hot encoded.
# By default, it expects a probability distribution
metric = {'label': CategoricalAccuracy('accuracy')}

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
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[callbacks],
    verbose=1)

# Evalueate on test data
model_eval = model.evaluate(
    x={'input_ids': np.array(x_test['input_ids'])},
    y={'label': np.array(y_test)}
)

# summarize history for accuracy
fig1 = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

# summarize history for loss
fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
fig1.savefig(f"accuracy_{timestamp}", dpi=300)
fig2.savefig(f"loss_{timestamp}", dpi=300)
model.save(f"model_{timestamp}")
