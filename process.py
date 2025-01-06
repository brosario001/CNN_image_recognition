import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments,moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.draw import rectangle_perimeter
import pickle

total_component_count = 0

def process_image(image_path, index, total_components_count, display_plots=True):
    #load image
    img = io.imread(image_path)

    #binirzation by thresholding
    threshold_value = 190
    img_binary = (img<threshold_value).astype(np.double)

    #connected component annalysis
    img_label = label(img_binary, background=0)

    #find number of components
    num_components = np.amax(img_label)
    print("The number of connected components: ", num_components)

    #display component bounding boxes
    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])   

    Features = []
    Labels = []
    th = 10

    for props in regions:
        minr, minc, maxr, maxc = props.bbox

        #ignore small size components
        if(maxr-minr)>th and (maxc-minc)>th:
            
            #red box and indexing
            ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, 
                                fill= False, edgecolor ='red', linewidth = 1))
            ax.text(minc, minr - 5, props.label+total_components_count, color='red', fontsize=10, ha='left', va='bottom')
            
            roi = img_binary[minr:maxr, minc:maxc]
            m = moments(roi)
            cc = m[0,1] / m[0,0]
            cr = m[1,0] / m[0,0]
            mu = moments_central(roi, center = (cr, cc))
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            Labels.append(ord(image_path[-5]) - ord('a') + 1)
            
    
    plt.savefig('data/figure'+str(index)+'.png')
    plt.close()
    
    
    return Features, Labels, num_components

        
