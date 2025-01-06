import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments,moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from normalize import normalize_features
from process import process_image

mean_values = np.loadtxt('data/mean_values.txt')
std_dev_values = np.loadtxt('data/std_dev_values.txt')

# Read the test image
test_image_path = 'img/test.bmp'
test_img = io.imread(test_image_path)

# Binarization (use the same threshold as in the training phase)
th = 190
test_img_binary = (test_img < th).astype(np.double)

# Connected Component Analysis
test_img_label = label(test_img_binary, background=0)

io.imshow(test_img_label)
io.show()

# Extracting Characters and Features
test_features = []

threshold_height = 10
threshold_width =10

io.imshow(test_img_binary)
ax = plt.gca()
ax.set_xticks([]) 
ax.set_yticks([])   

for props in regionprops(test_img_label):
    minr, minc, maxr, maxc = props.bbox
    roi = test_img_binary[minr:maxr, minc:maxc]

    # Check if the region is not too small (optional, depending on your specific requirements)
    if roi.shape[0] >= threshold_height and roi.shape[1] >= threshold_width:
        
        ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, 
                                fill= False, edgecolor ='red', linewidth = 1))
        ax.text(minc, minr - 5, props.label, color='red', fontsize=10, ha='left', va='bottom')
            
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)

        test_features.append(hu)

plt.savefig('data/test_indexing.png')
plt.close()


# Convert the list of features to a NumPy array
test_features_array = np.array(test_features)

# Normalize features using mean and variance from the training phase
normalized_test_features = (test_features_array - mean_values) / std_dev_values

# Load normalized features from the training phase
data = np.loadtxt('data/normalized_features.txt', delimiter=',')
normalized_training_features = data[:, :-1]  # Exclude the last column which contains labels
labels = data[:, -1]  # Extract labels

# Calculate the distance matrix between test and training features
D = cdist(normalized_test_features, normalized_training_features)

# Find the index of the closest match for each character in the test image
closest_matches = np.argmin(D, axis=1)

# Visualize the distance matrix (optional)
plt.imshow(D, cmap='viridis')
plt.title('Distance Matrix')
plt.show()


# Display the results or save them as needed
for i, closest_match in enumerate(closest_matches):
    print(f"Character {i + 1} in the test image is closest to character {closest_match + 1} in the training data.")
