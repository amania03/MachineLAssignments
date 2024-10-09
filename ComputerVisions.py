# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# 1. Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Display the shapes of the datasets
print("Shape of X_train:", X_train.shape)  # (60000, 28, 28)
print("Shape of X_test:", X_test.shape)    # (10000, 28, 28)
print("Shape of y_train:", y_train.shape)  # (60000,)
print("Shape of y_test:", y_test.shape)    # (10000,)
print("Shape of X_train[0]:", X_train[0].shape)  # (28, 28)
print("Shape of X_test[0]:", X_test[0].shape)    # (28, 28)

# 3. Visualize the first 5 images in the test set
np.set_printoptions(edgeitems=30, linewidth=100000)
for i in range(5):
    print("Label:", y_test[i], '\n')  # Print the label
    print("Matrix of values:\n", X_test[i], '\n')  # Print the matrix of values
    plt.contourf(np.rot90(X_test[i].transpose()))  # Make a contour plot of the matrix values
    plt.title(f"Digit: {y_test[i]}")
    plt.show()

# 4. Reshape the data
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Reshape to (60000, 784)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)    # Reshape to (10000, 784)

# 5. Initialize the kNN classifier (start with k = 3)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# 6. Fit the classifier on the training data
knn.fit(X_train_reshaped, y_train)

# 7. Predict on the test set
y_pred = knn.predict(X_test_reshaped)

# 8. Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with k={k}: {accuracy * 100:.2f}%")

# 9. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 10. Test with different values of k to find the optimal number of neighbors
accuracies = []
k_values = range(1, 20)  # Test for k from 1 to 20

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_reshaped, y_train)
    y_pred_k = knn.predict(X_test_reshaped)
    accuracies.append(accuracy_score(y_test, y_pred_k))

# Plot accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.grid()
plt.show()

#I used chatGPT, the solutions, and stack overflow to help me with this code. I also asked a TA a few questions.