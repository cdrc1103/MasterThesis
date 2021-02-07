# Load Huggingface transformer
from transformers import TFBertModel, BertConfig, BertTokenizerFast, TFTrainer, TFTrainingArguments

# Then what you need from tensorflow.keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# Libraries to import and process the data set
from Utilities import directories
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# Utilities
from Utilities.directories import lexis_abstract, savedModels
from contextlib import redirect_stdout
import pathlib

# Stuff that makes notebooks look nicer
import logging
from tensorflow import get_logger
get_logger().setLevel(logging.ERROR)


class Bert:
    def __init__(self, experiment_name, read_dir, save_dir, n_classes, max_token_length=100, model_name='bert-base-uncased'):
        self.experiment_name = experiment_name # name of the experiment
        self.read_dir = read_dir # where to read the data from
        self.save_dir = save_dir # where to save model weights and performance
        self.n_classes = n_classes
        self.max_token_length = max_token_length # number of tokens per example
        self.model_name = model_name

    def load_data(self):
        """

        :return:
        """

        train_files = tf.data.Dataset.list_files(file_pattern=str(self.read_dir) + "\*train*.tfrec")
        test_files = tf.data.Dataset.list_files(file_pattern=str(self.read_dir) + "\*test*.tfrec")
        validation_files = tf.data.Dataset.list_files(file_pattern=str(self.read_dir) + "\*validate*.tfrec")

        train_dataset = tf.data.TFRecordDataset(filenames=train_files, compression_type="ZLIB")
        test_dataset = tf.data.TFRecordDataset(filenames=test_files, compression_type="ZLIB")
        validation_dataset = tf.data.TFRecordDataset(filenames=validation_files, compression_type="ZLIB")

        features = {
            'abstract': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_function(example_proto, features):
            return tf.io.parse_single_example(example_proto, features)

        def select_data_from_record(record):
            x = record['abstract']
            y = record['label']
            return (x, y)


        train_dataset = train_dataset.map(lambda record: _parse_function(record, features))
        train_dataset = train_dataset.map(select_data_from_record)
        test_dataset = test_dataset.map(lambda record: _parse_function(record, features))
        test_dataset = test_dataset.map(select_data_from_record)
        validation_dataset = validation_dataset.map(lambda record: _parse_function(record, features))
        validation_dataset = validation_dataset.map(select_data_from_record)

        # Load transformers config and set output_hidden_states to False
        # from_pretrained loads weights from pretrained model
        config = BertConfig.from_pretrained(self.model_name)
        config.output_hidden_states = False
        print(config)
        with open(pathlib.Path.joinpath(self.save_dir, 'config.txt'), 'w') as file:
            with redirect_stdout(file):
                print(config)

        # Load the Transformers BERT model
        transformer_model = TFBertModel.from_pretrained(self.model_name, config=config) # loads all pretrained weights

        # Build your model input
        input_ids = Input(shape=(self.max_token_length,), name='input_ids', dtype='int64')
        inputs = {'input_ids': input_ids}

        # Load the MainLayer
        bert = transformer_model.layers[0]
        # Load the Transformers BERT model as a layer in a Keras model
        bert_model = bert(inputs)[1]
        dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
        pooled_output = dropout(bert_model, training=False)

        # Then build your model output
        label = Dense(units=self.n_classes,
                      kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
                      name='label')(
                pooled_output)
        outputs = {'label': label}

        # And combine it all in a model object
        model = Model(inputs=inputs, outputs=outputs, name=self.experiment_name)

        # Take a look at the model
        with open(pathlib.Path.joinpath(self.save_dir, 'summary.txt'), 'w') as file:
            # Pass the file handle in as a lambda function to make it callable
            print(model.summary(print_fn=lambda x: file.write(x + '\n')))

        plot_model(model, to_file=pathlib.Path.joinpath(self.save_dir, "model.png"))

        # Set an optimizer
        optimizer = Adam(
            learning_rate=5e-05,
            epsilon=1e-08,
            decay=0.01,
            clipnorm=1.0)

        # Set loss and metrics
        loss = {'label': CategoricalCrossentropy(
            from_logits=True)}  # Using from_logits=true means that the prediction tensor is
        # one hot encoded. By default, it expects a probability
        # distribution
        metric = {'label': CategoricalAccuracy('accuracy')}

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metric)

        # Fit the model
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            batch_size=16,
            epochs=10)

        model_eval = model.evaluate(
            test_dataset
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
        fig1.savefig(self.save_dir + r"\accuracy_" + timestamp, dpi=300)
        fig2.savefig(self.save_dir + r"\loss_" + timestamp, dpi=300)
        model.save(pathlib.Path.joinpath(self.save_dir, timestamp))


if __name__ == '__main__':
    experiment_name = "1.1_BertBaseUncased"
    read_dir = lexis_abstract
    save_dir = pathlib.Path.joinpath(savedModels, "1.1_BertBaseUncased")
    n_classes = 15
    b = Bert(experiment_name=experiment_name, read_dir=read_dir, save_dir=save_dir, n_classes=n_classes)
    b.load_data()