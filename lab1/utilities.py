import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)

    # Create a plot
    plt.figure()

    # Choose a color scheme for the plot 
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_confusion_matrix_values (y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def osypchuk_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values (y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def osypchuk_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values (y_true, y_pred)
    return (TP + TN) / (TP + FP + FN + TN)

def osypchuk_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def osypchuk_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_confusion_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def osypchuk_f1_score(y_true, y_pred):
    precision = osypchuk_precision_score(y_true, y_pred)
    recall = osypchuk_recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)