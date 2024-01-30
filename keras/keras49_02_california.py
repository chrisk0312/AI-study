from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense,LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

 