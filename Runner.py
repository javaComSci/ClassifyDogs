import numpy as np
import math
import pickle
from DataReader import DataReader 

if __name__ == "__main__":
    # create a reader and get all the images
    data_reader = DataReader("/train/n02099601-golden_retriever", "/train/n02113799-standard_poodle")
    data_reader.extract_data()

