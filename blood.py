import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import *
from keras import backend as K


# network and training

FEATURES = 4
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.2
VERBOSE = 0

LAYER = 2
N_HIDDEN = 5
DROPOUT = 0.0
EPOCHS = 10

MAX_LAYER = 10
MAX_N_HIDDEN = 30
MAX_DROPOUT = 0.3
MAX_EPOCHS = 50

while EPOCHS <= MAX_EPOCHS:
    LAYER = 2
    while LAYER <= MAX_LAYER:
        N_HIDDEN = 5
        while N_HIDDEN <= MAX_N_HIDDEN:
            DROPOUT = 0.0
            while DROPOUT <= MAX_DROPOUT:
                # Load Training Data
                training_data_df = pd.read_csv("blood.csv")

                X = training_data_df.drop('whether he/she donated blood in March 2007', axis=1).values
                Y = training_data_df[['whether he/she donated blood in March 2007']].values

                # Load the separate test data set
                test_data_df = pd.read_csv("bloodtest.csv")

                X_test = test_data_df.drop('whether he/she donated blood in March 2007', axis=1).values
                Y_test = test_data_df[['whether he/she donated blood in March 2007']].values

                K.clear_session()

                # Define the model
                model = Sequential()

                if LAYER == 1:
                    model.add(Dense(1, input_dim=FEATURES, kernel_initializer='uniform', activation='sigmoid'))

                else:
                    model.add(Dense(N_HIDDEN, input_dim=FEATURES, kernel_initializer='uniform', activation='relu'))
                    current_layers = 2

                    while current_layers < LAYER:
                        model.add(Dense(N_HIDDEN, kernel_initializer='uniform', activation='relu'))
                        model.add(Dropout(DROPOUT))
                        current_layers += 1

                    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                RUN_NAME = "La" + str(LAYER) + "No" + str(N_HIDDEN) + "Dr" + str(DROPOUT) + "Ep" + str(EPOCHS)

                # Create a TensorBoard logger
                logger = keras.callbacks.TensorBoard(
                    log_dir='logs/' + RUN_NAME,
                    histogram_freq=5,
                    write_graph=True
                )

                # Train the model
                model.fit(
                    X,
                    Y,
                    epochs=EPOCHS,
                    shuffle=True,
                    verbose=VERBOSE,
                    validation_split=VALIDATION_SPLIT,
                    callbacks=[logger]

                )

                score = model.evaluate(X_test, Y_test, verbose=0)
                print("---------------------------------")
                print("Epochs: " + str(EPOCHS) + "  Layer: " + str(LAYER) + "   Nodes: " + str(
                    N_HIDDEN) + "    Dropout: " + str(
                    DROPOUT))
                print("Test score/loss: ", score[0])
                print("Test accuracy: ", score[1])

                DROPOUT += 0.1

            N_HIDDEN += 5

        LAYER += 1

    EPOCHS += 10
