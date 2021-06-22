import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

class plotting:
    def plot_decision_boundary(model, X, y):
        """
        Plots the decision boundary created by a model predicting on X.
        This function has been adapted from two phenomenal resources:
        """
        # Define the axis boundaries of the plot and create a meshgrid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Create X values
        x_in = np.c_[xx.ravel(), yy.ravel()] # Stack 2D arrays together
        # Make predictions using the trained model
        y_pred = model.predict(x_in)

        # Check for multi-class
        if len(y_pred[0]) > 1:
            print("doing multiclass classification...")
            # Reshape our predictions to get them ready for plotting
            y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
        else:
            print("doing binary classifcation...")
            y_pred = np.round(y_pred).reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

    def plot_prediction_vs_data(X_train, X_test, y_train,y_test, predictions):
        """
        Plot the model's predictions against our regression data
        """
        plt.figure(figsize=(10, 7))
        plt.scatter(X_train, y_train, c='b', label='Training data')
        plt.scatter(X_test, y_test, c='g', label='Testing data')
        plt.scatter(X_test, predictions.squeeze(), c='r', label='Predictions')
        plt.legend();

    def plot_decision_boundary_test_and_train(model, X_train, X_test, y_train, y_test):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, X_train, y_train)
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, X_test, y_test)
        plt.show()

    def plot_history(history):
        pd.DataFrame(history.history).plot(figsize=(10,7), xlabel="epochs");

    def plot_learning_rate_vs_loss(history):
        plt.figure(figsize=(10, 7))
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss");

    
    

