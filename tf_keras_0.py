import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------------
def load_data(filename, shuffle=True,split_ratio=0.8):
    # Load Data
    df = pd.read_csv(filename,sep=',',header=0)
    labels = (df['y'].values + 1) / 2
    del df['y']
    points = df.values

    # Shuffle data
    if shuffle == True:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        points = np.array(points)[indices]
        labels = np.array(labels)[indices]

    # Divide the data
    N = int(split_ratio * len(labels))

    return points[:N], labels[:N], points[N:], labels[N:]

#------------------------------------------------------------------------------------
# Main Program
#------------------------------------------------------------------------------------
# Step 0: Global Parameters
epochs     = 100
lr_rate    = 0.01
batch_size = 32
model_name = "models/points_1_1/cp.ckpt"
data_path  = "data/data1_1.csv"
# data_path  = "data/data1_2.csv"
# data_path  = "data/data1_3.csv"
# data_path  = "data/data1_4.csv"
n_outputs  = 2

# Step 1: Create Model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(n_outputs,activation='sigmoid')])

# Step 2: Define Metrics
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = lr_rate),
              loss     = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics  = ['accuracy'])
# print(model.summary())

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_test, Y_test = load_data(data_path,True,0.8)

    # Step 4: Training
    # Create a function that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
                                                     save_weights_only=True,
                                                     verbose=0, save_freq=10)
    history = model.fit(X_train, Y_train,
                        batch_size     = batch_size,
                        epochs         = epochs,
                        validation_data= (X_test,Y_test),
                        callbacks      = [cp_callback])

    # Step 6: Evaluation
    loss,acc = model.evaluate(X_test, Y_test, verbose = 2)
    print("Evaluation, accuracy: {:5.2f}%".format(100 * acc))

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    input = [0,0]
    points = np.array([input])

    # Step 5: Predict the class
    preds = my_model.predict(points)
    print(np.argmax(preds[0]))
    print(preds)
