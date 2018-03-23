import numpy as np
from scipy.special import expit as sig
import matplotlib.pyplot as plt

def sigmoid(arr):
    return sig(arr)


def sigmoid_prime(arr):
    arr = sigmoid(arr)
    return arr*(1.0-arr)

def train(epoches,X,Y,weightHidden,weightOutput,lR):
    for epoch in range(epoches):
        hiddenIn = np.dot(X,weightHidden)
        hiddenOut = sigmoid(hiddenIn)
        outIn = np.dot(hiddenOut,weightOutput)
        outOut = outIn
        error = Y - outOut
        dErrorByDW2 = np.dot(hiddenOut.T,error*lR)
        weightOutput = weightOutput+dErrorByDW2
        dErrorByDW1 = np.dot(X.T,np.dot(error*lR,weightOutput.T)*sigmoid_prime(hiddenIn))
        weightHidden = weightHidden+dErrorByDW1
    return weightHidden,weightOutput


def test(testData,weightsHidden,weightsOutput):
    act_hidden = sigmoid(np.dot(testData, weightHidden))
    return (np.dot(act_hidden, weightOutput))





"""
X = np.array([[0,0],[0,1],[1,0],
[0.1,0.2],[0.1,0.4],[0.4,0.9],
[0.9,0],[0.99,0.99],[0.97,0.89],
[0.3,0.3],[0.89,0.78],[0.12,0.56]])

Y = np.array([[0],[1],[1],
[0],[0],[1],
[1],[1],[1],
[0],[1],[1]])

inputLayerSize, hiddenNeuronsSize, outputSize = 2, 3, 1

"""

X = []
Y = []

for i in range(50,80):
    X.append([i*1.0])
    Y.append([(i*1.8)+32.0])

X = np.array(X)
Y = np.array(Y)

inputLayerSize, hiddenNeuronsSize, outputSize = 1, 4, 1

epoches = 100000
lR = 0.001


weightHidden = np.random.uniform(size=(inputLayerSize, hiddenNeuronsSize))
weightOutput = np.random.uniform(size=(hiddenNeuronsSize, outputSize))

#weightHidden , weightOutput = train(epoches,X,Y,weightHidden,weightOutput,lR)


for epoch in range(epoches):
    hiddenIn = np.dot(X,weightHidden)
    hiddenOut = sigmoid(hiddenIn)
    outIn = np.dot(hiddenOut,weightOutput)
    outOut = outIn
    error = Y - outOut
    dErrorByDW2 = np.dot(hiddenOut.T,error*lR)
    weightOutput = weightOutput+dErrorByDW2
    dErrorByDW1 = np.dot(X.T,np.dot(error*lR,weightOutput.T)*sigmoid_prime(hiddenIn))
    weightHidden = weightHidden+dErrorByDW1


print (error)

output = test(np.array([69]),weightHidden,weightOutput)
print (output)
print ('expected ',(69*1.8)+32.0)
