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
import cv2

def convert_data_set_to_imgs(file_name):
    # make folder for visuals and load pickle
    foldername = file_name[:-7]+"-visualized/"
    os.makedirs(foldername, exist_ok =True)
    with gzip.open(file_name, "rb") as fp:
        X, Y = pickle.load(fp)

    length = len(X)
    for fridx, frame in enumerate(X):
        print(fridx, "out of", length)
        rgb = hsv_to_rgb(frame/255.0)
        plt.imsave(f"{foldername}/{fridx}-{'A' if Y[fridx] else 'B'}", rgb)

def seq_to_vid(vidpathwithoutfileformat, X, Y=None, fps=10):
    fontsize = 25
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", fontsize)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    clip = cv2.VideoWriter(f"{vidpathwithoutfileformat}.avi", fourcc, fps, (640,960))
    if Y is not None:
        plot = np.zeros((320,640,3), dtype=np.uint8)
        scalers = []
        for i in range(Y.shape[1]):
            minm = np.min(Y[:,i])
            maxm = np.max(Y[:,i])
            # if minm<0:
            #     maxm = -minm
            #     minm = 0
            #     Y[:,i] *= -1
            scalers.append((minm,(maxm-minm), 1 if maxm==0 else 0))
    for fridx, frame in enumerate(X):
        print(f"at frame {fridx} from {len(X)}", end='\r')
        if Y is not None:
            plot[:,:-5] = plot[:,5:]*0.9
            plot[plot<=0] = 0
            img = Image.fromarray(frame)
            img = img.resize((640,640))
            draw = ImageDraw.Draw(img)
            for rewidx in range(Y.shape[1]):
                value = Y[fridx,rewidx]
                x, y = 3, 1 + fontsize*rewidx
                draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=font)

                minm, scale, sign = scalers[rewidx]
                factor = (value-minm)/scale
                if sign:
                    factor = 1-factor
                plotvalue = 319-int(319*factor)
                plot[plotvalue-5:plotvalue+6, 640-(rewidx+1)*64-5:640-(rewidx+1)*64+6] = 255
            frame = np.array(img)
            frame = np.concatenate((frame, plot.astype(np.uint8)), axis=0)
            #plt.imshow(frame)
            #plt.show()
        clip.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    clip.release()

def uint8_color_converter(array, toRGB=True):
    if toRGB:
        return (255*hsv_to_rgb(array/255.0)).astype(np.uint8)
    else:
        return (255*rgb_to_hsv(array/255.0)).astype(np.uint8)

def data_set_to_vid(file_name, HSV=True, fps=10):
    foldername = file_name[:-7]+"-visualized/"
    os.makedirs(foldername, exist_ok =True)
    with gzip.open(file_name, "rb") as fp:
        X, Y = pickle.load(fp)
        X = X[::1]
        Y = Y[::1]
    if HSV:
        X = uint8_color_converter(X)
    if len(Y.shape)==1:
        Y = Y[:,None]

    seq_to_vid(file_name[:-7], X, Y=Y)

if __name__ == "__main__":
    #convert_data_set_to_imgs("/media/compute/homes/aharter/isy2020/minerl/data/split/tree-chop/split-HSV-ds10000-wait120-delay10-warmup20-chunk20/data.pickle")
    pass