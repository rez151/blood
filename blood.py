import pandas as pd
from keras.models import Sequential
from keras.layers import *

# network and training
LAYER = 1
N_HIDDEN = 100
FEATURES = 4
DROPOUT = 0.3
EPOCHS = 200
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
VERBOSE = 2

training_data_df = pd.read_csv("blood.csv")

X = training_data_df.drop('whether he/she donated blood in March 2007', axis=1).values
Y = training_data_df[['whether he/she donated blood in March 2007']].values

# print(X)
# print(Y)

# Define the model
model = Sequential()

if LAYER == 1:
    model.add(Dense(1, input_dim=FEATURES, kernel_initializer='uniform', activation='sigmoid'))
else:
    current_value = 0

    model.add(Dense(N_HIDDEN, input_dim=FEATURES, kernel_initializer='uniform', activation='relu'))

    while current_value < LAYER:
        model.add(Dense(N_HIDDEN, kernel_initializer='uniform', activation='relu'))


model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.add(Dense(100, input_dim=FEATURES, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(
    X,
    Y,
    epochs=EPOCHS,
    shuffle=True,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

# Load the separate test data set
test_data_df = pd.read_csv("bloodtest.csv")

X_test = test_data_df.drop('whether he/she donated blood in March 2007', axis=1).values
Y_test = test_data_df[['whether he/she donated blood in March 2007']].values

score = model.evaluate(X_test, Y_test, verbose=1)
print("Test score/loss: ", score[0])
print("Test accuracy: ", score[1])
