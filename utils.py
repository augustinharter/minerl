import torch as T
import torch.nn.functional as F
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import pickle
import minerl
import gzip

def convert_data_set_to_imgs(file_name):
    # make folder for visuals and load pickle
    foldername = file_name[:-7]+"-visualized/"
    os.makedirs(foldername, exist_ok =True)
    with gzip.open(file_name, "rb") as fp:
        X, Y = pickle.load(fp)

    for fridx, frame in enumerate(X):
        rgb = hsv_to_rgb(frame)
        plt.imsave(f"foldername/{fridx}-{'A' if Y[fridx] else 'B'}", rgb)

if __name__ == "__main__":
    convert_data_set_to_imgs()