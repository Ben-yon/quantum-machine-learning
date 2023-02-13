from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pennylane as qml
import numpy as np

iris = datasets.load_iris()
x = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

device = qml.device('default.qubit')


@qml.qnode(device)
def quantum_neural_net(weight, data):
    # initialize the qubits
    qml.templates.AmplitudeEmbedding(weight, data)
    
    # Apply a layer of quantum neurons
    qml.templates.StronglyEntanglingLayers(weight, data)
    
    # Measure the qubits
    return qml.expval(qml.PauliZ(0))


# at this point, we define the cost function
def cost(weights, data, labels):
    # make predictions using the quantum neural network 
    predictions = quantum_neural_net(weights, data)
    
    # Calculate the mean squared error
    mse = qml.mean_squared_error(labels, predictions)
    
    return mse


# Train the quantum model now
# Initialize the optimizer
opt = qml.AdamOptimizer(stepsize=0.01)

#  Set the number of training steps
steps = 100

# Set the initial weights
weights = np.random.normal(0, 1, (4, 2))

for i in range(steps):
    # Calculate the gradients
    gradient = qml.grad(cost, argnum=0)(weights, X_train_scaled, y_train)
    
    # Update the weights
    opt.step(gradient, weights)
