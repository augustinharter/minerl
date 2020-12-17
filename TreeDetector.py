import torch as T
import torch.nn.functional as F
from nets import Unet
import numpy as np
from matplotlib.colors import rgb_to_hsv
import pickle
import cv2
import sys

class TreeDetector():
    def __init__(self, modelpath="train/unet.pt",
        kmeanspath="train/kmeans.p", channels=3, HSV=True):
        self.toHSV = HSV
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.unet = Unet(colorchs=channels)
        self.unet.load_state_dict(
            T.load(modelpath, map_location=T.device(self.device)))
        if kmeanspath:
            with open(kmeanspath, "rb") as fp:
                #self.cluster, self.targetcluster = pickle.load(fp)
                self.cluster = pickle.load(fp)
        self.targetcluster = 2

    def convert(self, X):
        if len(X.shape) == 3:
            X = X[None]
        if type(X) == type(np.array([])):
            if np.max(X) > 1:
                X = X.astype(np.float) / 255
            X = T.from_numpy(X)
        elif X.max() > 1:
            X = X.float() / 255
        if self.toHSV:
            X = T.from_numpy(rgb_to_hsv(X)) * 255

        if self.cluster is not None:
            pixels = X.view(-1, 3)
            test = pixels.view(X.shape)
            assert (X == test).all()
            pixels = pixels[:, [0, 1]].float() / 255
            pixels[:, 1] *= 0.1

            flat_labels = self.cluster.predict(pixels)
            labels = flat_labels.reshape(X.shape[:-1])

        if X.shape[1] > X.shape[-1]:
            X = X.permute(0, 3, 1, 2)

        mask = self.unet(X.float()).detach().squeeze()
        raw_mask = mask.clone()

        if self.cluster is not None:
            label_mask = (labels != self.targetcluster)
            mask[label_mask] = 0

        cluster_mask = (label_mask==0)[0]
        return mask, raw_mask, cluster_mask

if __name__ == '__main__':
    # Detector Setup
    modelpath = "treecontroller/tree-control-stuff/unet.pt"
    kmeanspath = "treecontroller/tree-control-stuff/kmeans.pickle"
    
    blur = None
    if "blur3" in sys.argv:
        blurkernel = T.tensor([[[[1,2,1],[2,4,2], [1,2,1]]]*1]*3).float()/16
        blur = lambda x: F.conv2d(x, blurkernel, padding=1, groups=3)
        modelpath = "treecontroller/tree-control-stuff/unet-blur3.pt"
    if "blur5" in sys.argv:
        blurkernel = T.tensor([[[[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]]]*1]*3).float()/256
        blur = lambda x: F.conv2d(x, blurkernel, padding=2, groups=3)
        modelpath = "treecontroller/tree-control-stuff/unet-blur5.pt"

    detector = TreeDetector(modelpath, kmeanspath)


    # Video setup
    videopaths = [
        "live-clip-01.avi",
        "./data/MineRLTreechopVectorObf-v0/v3_agonizing_kale_tree_nymph-7_72884-74584/recording.mp4",
        #"./data/MineRLTreechopVectorObf-v0/v3_alarming_arugula_medusa-12_58066-60565/recording.mp4",
        #"./data/MineRLTreechopVectorObf-v0/v3_content_squash_angel-3_11240-12783/recording.mp4"
    ]
    videonames = ["live-01-segmented-new", "offline-01-segmented-new"]
    resultpath = "results/treedetect/"

    for vididx, videopath in enumerate(videopaths):
        cap = cv2.VideoCapture(videopath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(videonames[vididx]+'.avi',fourcc, 20.0, (64*3,64))
        #print(cap.attributes_)
 
        while(True):
            ret, BGR = cap.read()
            if not ret:
                break

            RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)

            # GENERATE SEGMENTATION FOR RGB FRAMES
            if blur is not None:
                blurred = blur(T.from_numpy(RGB)[None].float()).squeeze().numpy().astype(np.uint8)
                mask, _, _ = detector.convert(blurred)
            else:
                mask, _, _ = detector.convert(RGB)
                
            mask = mask.cpu().numpy()

            # Make visuals
            mask = np.stack((mask,mask,mask), axis=-1)
            gray = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)
            gray = np.stack((gray,gray,gray), axis=-1)
            overlay = (mask*BGR + (1-mask)*gray)
            combined = np.concatenate((BGR, overlay, 255*mask), axis=1).astype(np.uint8)
            #print(combined.shape)
            out.write(combined)
        #print(RGB.shape, np.max(RGB), RGB[0])
        cap.release()
        cv2.destroyAllWindows()
    