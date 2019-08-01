
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

URL = 'dataset_tf.csv'
dataframe = pd.read_csv(URL)

train, test = train_test_split(dataframe, test_size=0.4)
train, val = train_test_split(train, test_size=0.4)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

feature_columns = []

# numeric cols
for header in dataframe.columns:
   if header in ["id", "race", "sex", "target"]: continue
   feature_columns.append(feature_column.numeric_column(header))

# indicator cols
race = feature_column.categorical_column_with_vocabulary_list(
   "race", ["Amer Indian/Alaska Native", "Asian", "Black", "Multiple", "Native Hawaiian/Pacific Islander", "White"])
race_one_hot = feature_column.indicator_column(race)
feature_columns.append(race_one_hot)

sex = feature_column.categorical_column_with_vocabulary_list("sex", ["Male", "Female"])
sex_1hot = feature_column.indicator_column(sex)
feature_columns.append(sex_1hot)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
   dataframe = dataframe.copy()
   labels = dataframe.pop("target")
   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
   if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
   ds = ds.batch(batch_size)
   return ds


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
   feature_layer,
   tf.keras.layers.Dense(32, activation="relu"),
#   tf.keras.layers.dropout(0.5),
   tf.keras.layers.Dense(32, activation="relu"),
#   tf.keras.layers.dropout(0.5),
   tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=[
                  tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall(),
                  tf.keras.metrics.TruePositives(),
                  tf.keras.metrics.TrueNegatives(),
                  tf.keras.metrics.FalsePositives(),
                  tf.keras.metrics.FalseNegatives(),
                  tf.keras.metrics.AUC(),
              ],
              run_eagerly=False)

history = model.fit(train_ds, 
         validation_data=val_ds,
         epochs=20)

model.evaluate(test_ds)

precisions = [0] + history.history["precision"]
recalls = [0] + history.history["recall"]
aucs = [0] + history.history["auc"]
iterations = list(range(0, len(precisions)))
df_plot = pd.DataFrame({"iteration": iterations, "precision": precisions, "recall": recalls, "auc": aucs})
df_plot.set_index("iteration", inplace=True)
df_plot.plot(grid=True)
plt.show()


