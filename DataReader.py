import numpy as np
import math
import pickle
import os
import cv2
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import sklearn.cluster


# read all the data and pickle it
class DataReader:
    def __init__(self, folder_name_1, folder_name_2, folder_name_3):
        self.folder_name_1 = folder_name_1
        self.folder_name_2 = folder_name_2
        self.folder_name_3 = folder_name_3
    

    # read data from given folder
    def read_data(self, complete_paths):
        imgs = []

        # read in images
        for complete_path in complete_paths:
            for filename in os.listdir(complete_path):
                f = os.path.join(complete_path, filename)
                im = cv2.imread(f)
                if im is not None:
                    im = im.reshape(-1)
                    imgs.append(im)
        
        # create into numpy array
        all_imgs = np.asmatrix(imgs)
        # print(all_imgs[:100, :].shape)
        return all_imgs


    # extract data from images
    def extract_data(self):
        # build the complete paths of the folders
        complete_path_1 = os.getcwd() + self.folder_name_1
        complete_path_2 = os.getcwd() + self.folder_name_2
        complete_path_3 = os.getcwd() + self.folder_name_3

        # keep those images
        folder_images = self.read_data([complete_path_1, complete_path_2, complete_path_3])
        for i in range(folder_images.shape[0]):
            for j in range(folder_images.shape[1]):
                if folder_images[i,j] == 255:
                    folder_images[i,j] = 1

        pca = PCA(n_components=2)
        folder_images = pca.fit_transform(folder_images)
        
        km = sklearn.cluster.KMeans(n_clusters=3, max_iter=100)
        km.fit(folder_images)
        print(km.labels_)

        # print(folder_images.shape)
        
        # save all the images
        np.save("AllImgs.npy", folder_images)
        np.save("Folder1.npy", folder_images[:100, :])
        np.save("Folder2.npy", folder_images[100:, :])

                