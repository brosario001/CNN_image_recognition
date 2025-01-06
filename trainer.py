import numpy as np
import matplotlib.pyplot as plt
from normalize import normalize_features
from process import  process_image

image_paths = ['img/a.bmp','img/d.bmp','img/m.bmp','img/n.bmp','img/o.bmp','img/p.bmp','img/q.bmp','img/r.bmp','img/u.bmp','img/w.bmp',]

all_features=[]
all_labels=[]

i=0
total_component_count=0

for image_path in image_paths:
    features, labels, num_components = process_image(image_path,i,total_component_count)
    all_features.extend(features)
    all_labels.extend(labels)
    i = i+1
    total_component_count+=num_components

all_features_array = np.array(all_features)
all_labels_array = np.array(all_labels)

normalized_features, mean_values, std_dev_values = normalize_features(all_features)

np.savetxt('data/mean_values.txt', mean_values)
np.savetxt('data/std_dev_values.txt', std_dev_values)

data = np.column_stack((all_features, all_labels))
np.savetxt('data/normalized_features.txt', data, delimiter=',')
