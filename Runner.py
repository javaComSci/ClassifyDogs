import numpy as np
import math
import pickle
from DataReader import DataReader 
from KMeans import KMeans


if __name__ == "__main__":
    # create a reader and get all the images
    # data_reader = DataReader("/train/n02099601-golden_retriever", "/train/n02113799-standard_poodle")
    data_reader = DataReader("/shapes/circle", "/shapes/triangle", "/shapes/square")
    data_reader.extract_data()

    # with the data, extract the clusters given the number of clusters to do
    k_means_alg  = KMeans("AllImgs.npy", 3)
    k_means_alg.cluster()

