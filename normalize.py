import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments,moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

def normalize_features(features):
    features_array = np.array(features)
    mean_values = np.mean(features_array, axis=0)
    std_dev_values = np.std(features_array, axis = 0)
    normalized_features = (features_array - mean_values)/std_dev_values
    
    return normalized_features, mean_values, std_dev_values
    
    
    