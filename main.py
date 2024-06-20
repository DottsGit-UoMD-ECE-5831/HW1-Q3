"""
Q3) (SVD on your own group selfie!) With your ECE5831 teammates, take a selfie group photo. If
working remotely, you can put your individual selfies in a side-by-side manner and save it as a
single picture. Perform SVD matrix calculation on your group selfie and print the largest and
smallest singular values of your group photo. Then, plot all the singular values associated with your
group photo on a log-scale. Finally, try compressing the image of your group photo to the smallest
extent possible. Like the lecture material on SVD, start by retaining only 25% of values. Keep
reducing the size up to the point when you cannot see anything.

Author: Nicholas Butzke
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import rescale

# Function to perform SVD on an image
def svd_image_compression(image, k):
    # SVD
    U, S, V = np.linalg.svd(image, full_matrices=False)
    # Take top k singular values
    S = np.diag(S[:k])
    U = U[:, :k]
    V = V[:k, :]
    # Reconstruct image
    compressed_image = np.dot(U, np.dot(S, V))
    return compressed_image

def main():
    image_path = 'TeamPhoto.jpg'
    image = io.imread(image_path)
    image = color.rgb2gray(image)

    # SVD
    U, S, V = np.linalg.svd(image, full_matrices=False)

    # Print the largest and smallest singular values
    print(f"Largest singular value: {S[0]}")
    print(f"Smallest singular value: {S[-1]}")

    # Plot all singular values on a log scale
    plt.figure()
    plt.plot(np.log(S))
    plt.title('Singular values (log scale)')
    plt.xlabel('Index')
    plt.ylabel('Log(Singular value)')
    plt.show()

    # Compress the image by retaining different percentages of values
    percentages = [100, 25, 10, 5, 2.5, 1, 0.1]
    plt.figure(figsize=(25, 10))
    for i, p in enumerate(percentages):
        k = int(len(S) * p / 100)
        compressed_image = svd_image_compression(image, k)
        plt.subplot(1, len(percentages), i + 1)
        plt.imshow(compressed_image, cmap='gray')
        plt.title(f"{p}% values")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()