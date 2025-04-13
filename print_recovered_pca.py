import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from tensorflow.keras.datasets import cifar10  # Import CIFAR-10 dataset from Keras
import os

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

# 3. Set k = 40
k = 40
print(f"Running PCA with k = {k}...")

# 4. Perform Incremental PCA with k = 40
ipca = IncrementalPCA(n_components=k, batch_size=500)  # Use mini-batch size of 500

# Perform PCA transformation to reduce the data to k components
X_reduced = ipca.fit_transform(X_centered)  # Apply Incremental PCA and reduce the dimensionality of the data

# Reconstruct the data from the reduced representation
X_reconstructed = ipca.inverse_transform(X_reduced)  # Inverse transform to get the data back to original space

# Create output directory to save images
output_dir = "reconstructed_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the original and reconstructed images for the first 10 images
for i in range(20):
    # Reconstruct the image
    reconstructed_image = X_reconstructed[i].reshape(32, 32, 3)  # Reshape to original image shape
    reconstructed_image += mean.reshape(32, 32, 3)  # Add the mean to the reconstructed image
   
    # Create a comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row and 2 columns for original and reconstructed images
    axes[0].imshow(X_train[i].reshape(32, 32, 3).astype(np.uint8))
    axes[0].set_title(f"Data", fontsize=30)  # Set title for original image with larger font size
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image.astype(np.uint8))
    axes[1].set_title(f"Reconstructed", fontsize=30)  # Set title for reconstructed image with larger font size
    axes[1].axis('off')

    # Save the image comparison pair as a single file
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"image_comparison_{i+1}.png"))
    plt.close()  # Close the plot to avoid it being shown in the notebook

print(f"Images saved in {output_dir} directory.")