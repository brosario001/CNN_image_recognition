import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments,moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from normalize import normalize_features

data = np.loadtxt('data/normalized_features.txt', delimiter=',')
normalized_features = data[:, :-1]
labels = data[:, -1]

D = cdist(normalized_features, normalized_features)
D_index = np.argsort(D, axis=1)
predicted_labels = labels[D_index[:, 1]]
confusion_matrix_result = confusion_matrix(labels, predicted_labels)

plt.imshow(D, cmap='viridis')
plt.title('Distance Matrix')
plt.show()

plt.imshow(confusion_matrix_result, cmap='viridis', interpolation='nearest')
plt.title('Confusion Matrix')
plt.show()