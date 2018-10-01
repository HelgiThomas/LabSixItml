from sklearn.datasets import load_digits 

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import robust_scale, scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

# #############################################################################
# I: Load and Describe the Data

print(digits.keys())
def describeDataset (data):
      print("digits: %d, \nsamples: %d, \nfeatures: %d \nlabels: %s"
      % (n_digits, n_samples, n_features, labels))
      # printImages(digits.data[:5], ["Training: " + str(x) for x in digits.target[:5]])

def printImages(images, labels):
      plt.figure(figsize=(20,4))
      for index, (image, label) in enumerate(zip(images, labels)):
            plt.subplot(1, 5, index + 1)
            plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
            plt.title(label, fontsize = 20)
      plt.show()

describeDataset(data)

# #############################################################################
# II: Cluster the Data
print(82 * '_')

kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(data)


# #############################################################################
# III: Visualize the data
print(82 * '_')

# ##################################
# III (a): Print defining images
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit_predict(data)

centroids = kmeans.cluster_centers_

# centroIDs = [index for index, point in enumerate(data) if point in centroids]

for centroid in centroids:
      print(centroid)

# ##################################
# III (b): Dimensionality Reduction
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()