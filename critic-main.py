import argparse
from matplotlib import pyplot as plt
import utils
import os
from nets import AutoEncoder, VAE, Critic
import numpy as np
import torch as T
import torch.nn.functional as F
import pickle
import math
import minerl
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib import cm
import io
import cv2
from sklearn.cluster import KMeans
import logging as L


class Handler():
    def __init__(self, args):
        #os.environ["MINERL_DATA_ROOT"] = "/home/augo/uni/minerl/data"
        #self.data = minerl.data.make('MineRLTreechopVectorObf-v0')
        self.args = args
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        self.AE = Critic
        self.models = dict()
        self.models["auto-encoder"] = self.AE
        self.arg_path = f"default/"
        print("model path:", self.arg_path)
        self.result_path = f"./results/Critic/"+ args.name+ "-"+ self.arg_path
        print("viz path:", self.result_path)
        self.save_path = f"./saves/Critic/"+self.arg_path

    def load_data(self, data_size = 10000, batch_size = 8):
        data_path = f"data/tiles/tree-chop/default/"
        file_name = "data.pickle"
        test_size = 1000

        print("loading data:", data_path)
        # TRAIN
        if not os.path.exists(data_path+file_name):
            print("train set...")
            os.makedirs(data_path, exist_ok=True)
            self.collect_dataset(data_path+file_name, size=data_size)
        # TEST
        if not os.path.exists(data_path+"test-"+file_name):
            print("collecting test set...")
            os.makedirs(data_path, exist_ok=True)
            self.collect_dataset(data_path+"test-"+file_name, size=test_size)

        # TRAIN data
        with open(data_path+file_name, "rb") as fp:
            self.X, self.Y = pickle.load(fp)
        self.dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.X), T.from_numpy(self.Y)), batch_size=32, shuffle=True)
        print(f"loaded train set with {len(self.data)}")

        self.tiles_per_frame = len(self.data[0])

        # TEST data
        with open(data_path+"test-"+file_name, "rb") as fp:
            self.XX, self.YY = pickle.load(fp)
        self.testdataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.XX), T.from_numpy(self.YY)), batch_size=32, shuffle=False)
        print(f"loaded test set with {len(self.XX)}")

    def load_models(self):
        for model in self.models:
            print("loading:", self.save_path+f'{model}.pt')
            self.models[model].load_state_dict(T.load(self.save_path+f'{model}.pt', map_location=T.device(self.device)))

    def save_models(self):
        os.makedirs(self.save_path, exist_ok=True)
        save_path = self.save_path
        for model in self.models:
            print("saving:", save_path+f'{model}.pt')
            T.save(self.models[model].state_dict(), save_path+f'/{model}.pt')

    def forward(self, mode="train"):
        data = self.data
        args = self.args
        testf = mode=="test"
        trainf = mode=="train"
        loader = self.dataloader if trainf else self.testdataloader

        # Setup save path and Logger
        result_path = self.result_path+mode+"/" 
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        critic = self.critic
        opti = T.optim.Adam(critic.parameters())
        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            for b_idx, (X,Y) in enumerate(loader):

                # FORWARD PASS---------------------------
                if testf: # Stop early if testing
                    if b_idx>=10:
                        break

                pred = critic(X)
                if trainf:
                    loss = F.mse_loss(pred, y)
                

                print(log_msg, end="\r")
                log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx%100) or testf: # VISUALIZE
                    plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", diff)
            
            if epoch and not epoch%args.saveevery:
                self.save_models()
            
            if trainf:
                #critic.sched.step()
                pass
 
        print()
        log_file.close()

    def collect_dataset(self, path, size=2000, cons=50):
        os.environ["MINERL_DATA_ROOT"] = "/home/augo/uni/minerl/data"
        data = minerl.data.make('MineRLTreechopVectorObf-v0', num_workers=1, worker_batch_size=1)
        X = []
        Y = []
        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(10,100, num_epochs=1, preload_buffer_size=1)):
            print("at batch", b_idx, end='\r')
            #vector = state['vector']

            # CONVERt COLOR
            pov = state['pov']/255
            if self.args.color == "HSV":
                pov = rgb_to_hsv(pov)

            gamma = 0.9
            stepsize = 2
            wait = 5
            #chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>wait+stepsize*cons]
            chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>wait]
            approaches = []
            rewards = []
            for row, frame in chops:
                #approaches.append(pov[row,frame-wait-stepsize*cons : frame-wait : stepsize]) # take 30 frames from 50 frames before chop
                approaches.append(pov[row, 0: frame-wait : stepsize])
                rawrew = rew[row, 0: frame-wait : stepsize]
                fak = 1
                for i in range(1, len(rawrew)+1):
                    rawrew[-i] = fak
                    fak *= gamma
                rewards.append(rawrew)
            
            #print(approaches)
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            print(len(X))

            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        X = np.array(X)
        Y = np.array(Y)
        with open(path, "wb") as fp:
            pickle.dump((X, Y), fp)

    def get_arr_from_fig(self, fig, dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64,64))
        return img/255

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("--gray", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--color", type=str, default="HSV")
    parser.add_argument("--name", type=str, default="default")
    args = parser.parse_args()
    print(args)

    H = Handler(args)
    H.load_data()

    if args.load:
        H.load_models()
    if args.train:
        H.forward(mode="train")
        H.save_models()
    if args.test:
        if not args.train:
            H.load_models()
        H.forward(mode="test")
    
    exit(0)
