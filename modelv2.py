import keras as k
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import IPythonConsole
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import math
import random
import collections

# altering to use a % of the input data for test
fraction_of_test_molecules=0.1
validation_split=0.1
class_to_charge_list=[0,1,-1,-2]

"""TRAINING DATA PREPROCESSING"""
#Convert training data, in the form of txt files of line-by-line SMILES strings and charges into arrays
with open('1K_SMILES_strings.txt') as my_file:
    SMILES_array = my_file.readlines()

with open('1K_SMILES_strings_charges.txt') as my_file:
    charges_array = my_file.readlines()

no_of_data_points=int(len(charges_array))
no_of_test_points=int(fraction_of_test_molecules*no_of_data_points)
print('{} datapoints, {} randomly selected for test, leaving {} for train'.format(no_of_data_points, no_of_test_points, no_of_data_points-no_of_test_points))


test_line_nos=random.sample(range(0, no_of_data_points), no_of_test_points)
train_line_nos=[x for x in range(0, no_of_data_points) if x not in test_line_nos]

no_of_train_points=int(len(train_line_nos))
print('Validation split is {}, so {} points val, {} points pure train'.format(validation_split, int(validation_split*no_of_train_points), int(no_of_train_points-(validation_split*no_of_train_points))))
# make new arrays
X_train = []
T_train = []
for i in train_line_nos:
    X_train.append(SMILES_array[i])
    T_train.append(charges_array[i])


X_test = []
T_test = []
for i in test_line_nos:
    X_test.append(SMILES_array[i])
    T_test.append(charges_array[i])

print('Randomly assigned test set set up: stats follows:')
print('Train: {}'.format(collections.Counter(T_train)))
print('Test: {}'.format(collections.Counter(T_test)))


#Convert testing data, in the form of txt file of line-by-line SMILES strings into arrays
#with open('10_SMILES_strings_test.txt') as my_file:
#    test_SMILES_array = my_file.readlines()

#Convert each item of the training array of SMILES strings into molecules
mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in X_train]

#Convert training molecules into training fingerprints
bi = {}
fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
np_fps_array = []
for fp in fps:
  arr = np.zeros((1,), dtype= int)
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_fps_array.append(arr)

"""TESTING DATA PREPROCESSING"""
#Convert each item of the testing array of SMILES strings into molecules
test_mols = [Chem.rdmolfiles.MolFromSmiles(test_SMILES_string) for test_SMILES_string in X_test]

#Convert testing molecules into testing fingerprints
test_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(test_m, radius=2, nBits=256) for test_m in test_mols]

#Convert testing fingerprints into binary, and put all testing binaries into arrays
test_np_fps_array = []
for test_fp in test_fps:
  test_arr = np.zeros((1,), dtype= int)
  DataStructs.ConvertToNumpyArray(test_fp, test_arr)
  test_np_fps_array.append(test_arr)

"""NEURAL NETWORK"""
#The neural network model
model = Sequential([
    Dense(256, input_shape=(256,), activation= "relu"),
    Dense(128, activation= "sigmoid"),
    Dense(64, activation= "sigmoid"),
    Dense(34, activation= "sigmoid"),
    Dense(16, activation= "sigmoid"),
    BatchNormalization(axis=1),
    Dense(4, activation= "softmax")
])
model.summary()

#Compiling the model
model.compile(optimizer=Adam(lr=0.00001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Training the model
model.fit(np.array(np_fps_array), np.array(T_train), validation_split=validation_split, batch_size=10, epochs= 100, shuffle=True, verbose=1)

#Predictions with test dataset
predictions = model.predict(np.array(test_np_fps_array), batch_size=1, verbose=1)
no_correct = 0

print('class\t0\t1\t\t2\t\t3')
print('charge\t0\t+1\t\t-1\t\t-2')

for index in range(len(predictions)):
    prediction=predictions[index]
    print (prediction)
    predicted_class=np.argmax(prediction)    
    ground_truth_class=int(T_test[index][0])
    print('Predicted class {}, ground truth class {}'.format(predicted_class, ground_truth_class))
    print('Predicted charge {}, ground truth charge {}'.format(
        class_to_charge_list[predicted_class], 
        class_to_charge_list[ground_truth_class]))
    if predicted_class == ground_truth_class:
        no_correct = no_correct+1
        print('Correct')
    else:
        print('Incorrect')

print('{} correct, {}%'.format(no_correct, 100*(no_correct/no_of_test_points)))
