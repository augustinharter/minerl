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
import io

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

def get_arr_from_fig(fig, dpi=100, size=(256,256)):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def seq_to_vid(vidpathwithoutfileformat, X, Y=None, fps=10, detailed=True, norm_plot=True, norm_text=False):
    plt.ion()
    fontsize = 25
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", fontsize)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    clip = cv2.VideoWriter(f"{vidpathwithoutfileformat}-new.avi", fourcc, fps, (640,960))

    if Y is not None:
        if detailed:
            fig = plt.figure(figsize=(8,4))
            ax = fig.subplots()
        else:
            #ax = fig.subplots()
            #fig = plt.figure(figsize=(100,4))
            pass

        label = ["rel chop idx", "exp decay reset", "exp decay add", "rev decay reset", "rev decay add", "subtract reset"]

        YY = Y.copy()
        if norm_plot:
            for i in range(Y.shape[1]):
                minm = np.min(Y[:,i])
                maxm = np.max(Y[:,i])

                Y[:,i] = (Y[:,i]-minm)/(maxm-minm)
                #ax.plot(Y[:,i])
                ax.plot(Y[:,i], label=label[i])
        if norm_text:
            YY = Y

        if not detailed:
            #complete = get_arr_from_fig(fig, size=(8000,320))[:,80:-80]
            pass
        else:
            ax.plot([0,0], [0,1])
            ax.legend()
            ax.set_xlabel("time step")
            ax.set_ylabel("normalized reward")
        #plt.show()

    for fridx, frame in enumerate(X):
        print(f"at frame {fridx} from {len(X)}", end='\n')
        if Y is not None:
            #if not detailed:
            #    xpos = int(fridx*complete.shape[1]/len(Y))
            #    xstart = xpos-320
            #    plot = complete[:, xstart:xstart+640].copy()
            #    if plot.shape[1]<640:
            #       continue
            #    plot[:,plot.shape[1]//2] = 0

            img = Image.fromarray(frame)
            img = img.resize((640,640))
            draw = ImageDraw.Draw(img)
            for rewidx in range(Y.shape[1]):
                value = YY[fridx,rewidx]
                x, y = 3, 1 + fontsize*rewidx
                draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=font)

                if detailed:
                    values = Y[max(fridx-100, 0):fridx+100,rewidx]
                    ax.lines[rewidx].set_ydata(values)
                    ax.lines[rewidx].set_xdata(np.arange(max(fridx-100, 0),max(fridx-100, 0)+len(values)))
            if detailed:
                ax.lines[-1].set_xdata([fridx,fridx])
                ax.set_xlim(max(fridx-100, 0),fridx+100)
                plot = get_arr_from_fig(fig, size=(640,320))
            

            frame = np.array(img)
            frame = np.concatenate((frame, plot), axis=0)
            #plt.imshow(frame)
            #plt.show()
        clip.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #if fridx>100:
        #    break
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