import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from tensorflow.keras.datasets import cifar10  # Import CIFAR-10 dataset from Keras

# 1. Load the CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (_, _) = cifar10.load_data()  # X_train contains the images, y_train contains the labels
print("CIFAR-10 dataset loaded.")

# 2. Preprocess the data
print("Preprocessing the data...")

# Use the entire CIFAR-10 dataset (50,000 images)
X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float64)  # Flatten each image to a 1D vector (3072 features)

# Normalize the dataset by subtracting the mean (centering the data).
# This is necessary because PCA works best when the data is centered (mean is 0).
mean = np.mean(X_train, axis=0)  # Calculate the mean of the training data across all images (mean of each pixel across all images)
X_centered = X_train - mean  # Subtract the mean from each image to center the data

print("Data preprocessing complete.")

# 3. Set up the range of k values to test (k from 1 to 50)
k_values = range(1, 101)  # Test k values from 1 to 50 components

# 4. Initialize a list to store reconstruction errors for each k
errors = []  # List to store the reconstruction error for each k

# 5. Perform Incremental PCA for each k value and compute reconstruction error
print("Starting Incremental PCA computations...")
for i, k in enumerate(k_values):
    # Create an IncrementalPCA model with k components
    ipca = IncrementalPCA(n_components=k, batch_size=500)  # Use mini-batch size of 500
    
    # Perform PCA transformation to reduce the data to k components
    X_reduced = ipca.fit_transform(X_centered)  # Apply Incremental PCA and reduce the dimensionality of the data
    
    # Reconstruct the data from the reduced representation (this is how we get back the original data from reduced form)
    X_reconstructed = ipca.inverse_transform(X_reduced)  # Inverse transform to get the data back to original space
    
    # Compute the reconstruction error (Mean Squared Error)
    reconstruction_error = np.mean((X_centered - X_reconstructed) ** 2)  # MSE between original and reconstructed data
    
    # Append the reconstruction error for this value of k to the list of errors
    errors.append(reconstruction_error)
    
    # Print the error for the current k
    print(f"k = {k}, Reconstruction Error (MSE) = {reconstruction_error:.6f}")

print("Incremental PCA computations complete.")

# 6. Plot the reconstruction error as a function of k
print("Plotting the reconstruction error vs. number of components...")
plt.plot(k_values, errors, marker='o', markersize=3)  # Reduce the dot size (markersize=3)
plt.xlabel('Number of Components (k)')  # Label for the x-axis
plt.ylabel('Reconstruction Error (MSE)')  # Label for the y-axis
plt.title('Reconstruction Error vs Number of Components (k)')  # Title for the plot
plt.grid(True)  # Show gridlines on the plot to make it easier to read
plt.show()  # Display the plot

# 7. Find the optimal k value
# The optimal k is the value that minimizes the reconstruction error.
# np.argmin(errors) returns the index of the minimum error value.
optimal_k = k_values[np.argmin(errors)]  # Find the k value corresponding to the minimum reconstruction error
print(f"The optimal number of components is: {optimal_k}")  # Output the optimal k
