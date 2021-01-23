# Load Huggingface transformers
from transformers import TFBertModel, BertConfig, BertTokenizerFast
# Then what you need from tensorflow.keras
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

# Stuff that makes the notebook look nicer
import logging
from tensorflow import get_logger
get_logger().setLevel(logging.ERROR)

