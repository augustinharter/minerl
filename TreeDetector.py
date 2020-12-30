import torch as T
import torch.nn.functional as F
from nets import Unet, GroundedUnet
import numpy as np
from matplotlib.colors import rgb_to_hsv
import pickle
import cv2
import sys
from PatchEmbedder import PatchEmbedder
import os

class UnetTreeDetector():
    def __init__(self, modelpath="train/unet.pt",
        kmeanspath="train/kmeans.p", channels=3, HSV=True, grounded=False, blur=0):
        self.toHSV = HSV
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        if grounded:
            self.unet = GroundedUnet(colorchs=channels)
        else:
            self.unet = Unet(colorchs=channels)
        self.unet.load_state_dict(
            T.load(modelpath, map_location=T.device(self.device)))
        if kmeanspath:
            with open(kmeanspath, "rb") as fp:
                #self.cluster, self.targetcluster = pickle.load(fp)
                self.cluster = pickle.load(fp)
        self.targetcluster = 2

        self.blur = None
        if blur==5:
            blurkernel = T.tensor([[[[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]]]*1]*3).float()/256
            self.blur = lambda x: F.conv2d(x, blurkernel, padding=2, groups=3)
        if blur==3:
            blurkernel = T.tensor([[[[1,2,1],[2,4,2], [1,2,1]]]*1]*3).float()/16
            self.blur = lambda x: F.conv2d(x, blurkernel, padding=1, groups=3)

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
        else:
            X *= 255

        if self.blur is not None:
            #blurred = np.swapaxes(RGB,0,-1)
            #blurred = blur(T.from_numpy(blurred)[None].float())[0].numpy().astype(np.uint8)
            X = self.blur(X)
            #blurred = np.swapaxes(blurred,0,-1)

        if self.cluster is not None:
            #print(X.shape)
            pixels = X.reshape(-1, 3)
            test = pixels.reshape(X.shape)
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

class PatchEmbedTreeDetector():
    def __init__(self, embed_tuple_path="train/embed-data.pickle"):
        super().__init__()
        if not os.path.exists(embed_tuple_path):
            print("ERROR no embed tuple found!")
            exit()
        else:
            print("found embed tuple, now initializing PatchEmbedder...")
            self.embedder = PatchEmbedder()
            self.embedder.load_embed_tuple(embed_tuple_path)
        
        self.thresh = 0.7

    def detect(self, X):
        # MAKE PATCHES
        if len(X.shape) == 3:
            X = X[None]
        elif X.shape[0] != 1:
            print("WRONG INPUT SHAPE:", X.shape, "expected one rgb image WxWx3")
            exit(1)
        X = X/255.0
        size = X.shape[1:3]
        X = rgb_to_hsv(X)


        # CALC PROBS
        #patches = self.embedder.make_patches(X, 8, 2)
        #probs = self.embedder.calc_tree_probs_for_patches(patches)
        probs = self.embedder.predict_batch(X)
        return cv2.resize(probs[0], size)

    def get_tree_locations(self, X):
        probs = self.detect(X)

        thresh_probs = probs > self.thresh

        # FIND COMPONENTS
        number, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_probs.astype(np.uint8))
        order = np.argsort(stats[:,cv2.CC_STAT_AREA],)[::-1][:5]

        return probs, centroids[order], stats[order, cv2.CC_STAT_AREA]



if __name__ == '__main__':
    # Detector Setup
    unetpath = "treecontroller/tree-control-stuff/unet.pt"
    kmeanspath = "treecontroller/tree-control-stuff/kmeans.pickle"
    
    if "-blur3" in sys.argv:
        unetpath = "treecontroller/tree-control-stuff/unet-blur3.pt"
    if "-blur5" in sys.argv:
        unetpath = "treecontroller/tree-control-stuff/unet-blur5.pt"
    if "-grounded" in sys.argv:
        unetpath = "treecontroller/tree-control-stuff/unet-grounded.pt"

    #detector = UnetTreeDetector(unetpath, kmeanspath, grounded="-grounded" in sys.argv, HSV=not "-resnet" in sys.argv)
    detector = PatchEmbedTreeDetector()

    # Video setup
    videopaths = [
        "./eval/live-clip-01.avi",
        "./data/MineRLTreechopVectorObf-v0/v3_agonizing_kale_tree_nymph-7_72884-74584/recording.mp4",
        #"./data/MineRLTreechopVectorObf-v0/v3_alarming_arugula_medusa-12_58066-60565/recording.mp4",
        #"./data/MineRLTreechopVectorObf-v0/v3_content_squash_angel-3_11240-12783/recording.mp4"
    ]
    videonames = [f"live-01-patch-embed-integrated", f"offline-01-patch-embed-integrated"]
    resultpath = "results/treedetect/"

    frameidx = 0
    for vididx, videopath in enumerate(videopaths):
        cap = cv2.VideoCapture(videopath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("./eval/"+videonames[vididx]+'.avi',fourcc, 20.0, (64*4,64))
        #print(cap.attributes_)
 
        while(True):
            ret, BGR = cap.read()
            if not ret:
                break

            RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
            print("at frame", frameidx, end='\r')
            frameidx += 1

            # GENERATE SEGMENTATION FOR RGB FRAMES
            mask, centers, centersizes = detector.get_tree_locations(RGB)
            #print(centers[1], centersizes[1])
            if len(centers)>=2:
                #print(centers.shape)
                nav_point = np.round(centers[1]).astype(int)
                #print(nav_point)
                BGR[nav_point[1], nav_point[0]] = 1
                
            #mask = mask.cpu().numpy()

            # Make visuals
            mask = np.stack((mask,mask,mask), axis=-1)
            clean_mask = mask.copy()
            clean_mask[clean_mask<0.8] = 0
            gray = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)
            gray = np.stack((gray,gray,gray), axis=-1)
            overlay = (mask*BGR + (1-mask)*gray)
            combined = np.concatenate((BGR, overlay, 255*mask, 255*clean_mask), axis=1).astype(np.uint8)
            #print(combined.shape)
            out.write(combined)
        #print(RGB.shape, np.max(RGB), RGB[0])
        cap.release()
        cv2.destroyAllWindows()
    