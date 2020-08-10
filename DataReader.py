import numpy as np
import math
import pickle
import os
import cv2


# read all the data and pickle it
class DataReader:
    def __init__(self, folder_name_1, folder_name_2):
        self.folder_name_1 = folder_name_1
        self.folder_name_2 = folder_name_2
    

    # read data from given folder
    def read_data(self, complete_paths):
        imgs = []

        # read in images
        for complete_path in complete_paths:
            for filename in os.listdir(complete_path):
                f = os.path.join(complete_path, filename)
                im = cv2.imread(f)
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

        # keep those images
        folder_images = self.read_data([complete_path_1, complete_path_2])
        
        # save all the images
        np.save("AllImgs.npy", folder_images)
        np.save("Folder1.npy", folder_images[:100, :])
        np.save("Folder2.npy", folder_images[100:, :])

                