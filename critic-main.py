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
import gzip
from PIL import Image, ImageDraw, ImageFont


class Handler():
    def __init__(self, args):
        #os.environ["MINERL_DATA_ROOT"] = "data"
        #self.data = minerl.data.make('MineRLTreechopVectorObf-v0')
        self.args = args
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        print("device:", self.device)
        self.critic = Critic().to(self.device)
        self.models = dict()
        self.models["auto-encoder"] = self.critic
        self.arg_path = f"wait{args.wait}/"
        print("model path:", self.arg_path)
        self.result_path = f"./results/Critic/"+ args.name+ "-"+ self.arg_path
        print("viz path:", self.result_path)
        self.save_path = f"./saves/Critic/"+self.arg_path

        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 9)
        
        L.basicConfig(filename=f'logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def load_data(self, batch_size = 32):
        wait = self.args.wait
        test_size = 600
        data_size = self.args.datasize
        data_path = f"data/split/tree-chop/{self.args.color}-ds{data_size}-wait{wait}-delay{self.args.delay}-warm{self.args.warmup}-g{self.args.gamma}-rg{self.args.revgamma}-cons{self.args.cons}/"
        file_name = "data.pickle"

        print("loading data:", data_path)
        # TRAIN
        #if not os.path.exists(data_path+file_name):
        #    print("train set...")
        #    os.makedirs(data_path, exist_ok=True)
        #    self.collect_dataset(data_path+file_name, size=data_size, wait=wait, datadir=data_path)
        # TEST
        if not os.path.exists(data_path+"test-"+file_name):
            print("collecting test set...")
            os.makedirs(data_path, exist_ok=True)
            self.collect_dataset(data_path+"test-"+file_name, size=test_size, wait=wait, datadir=data_path)

        # TRAIN data
        #with gzip.open(data_path+file_name, "rb") as fp:
            self.X, self.Y = pickle.load(fp)
        #self.dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.X), T.from_numpy(self.Y)), batch_size=batch_size, shuffle=True)
        #print(f"loaded train set with {len(self.X)}")

        # TEST data
        with gzip.open(data_path+"test-"+file_name, "rb") as fp:
            self.XX, self.YY = pickle.load(fp)
        self.testdataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.XX), T.from_numpy(self.YY)), batch_size=batch_size, shuffle=False)
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

                XP = X.permute(0,3,1,2).float().to(self.device)
                Y = Y[:,args.rewidx].float().to(self.device)
                #print(X.shape, Y.shape, Y)

                pred = critic(XP).squeeze()
                if trainf:
                    loss = F.mse_loss(pred, Y)
                
                if trainf:
                    print(loss.item(), end="\r")
                #log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx%100) or testf: # VISUALIZE
                    if trainf:
                        L.info(f"e{epoch} b{b_idx} loss: {loss.item()}")
                        order1 = pred.argsort(descending=True)
                        order2 = Y.argsort(descending=True)
                    if testf:
                        order1 = np.arange(Y.shape[0])
                        order2 = np.arange(Y.shape[0])
                    viz = hsv_to_rgb(X[order1].numpy()/255) if self.args.color == "HSV" else X[order].numpy()/255
                    viz = np.concatenate(viz, axis=1)
                    viz2 = hsv_to_rgb(X[order2].numpy()/255) if self.args.color == "HSV" else X[order].numpy()/255
                    viz2 = np.concatenate(viz2, axis=1)

                    viz = np.concatenate((viz,viz2), axis=0)
                    img = Image.fromarray(np.uint8(255*viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(pred[order1].tolist()):
                        x, y = int(i*img.width/len(pred)), 1
                        draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)
                    for i, value in enumerate(Y[order2].tolist()):
                        x, y = int(i*img.width/len(Y)), int(1+img.height/2)
                        draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)

                    #plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path+f"e{epoch}_b{b_idx}.png")
            
            if epoch and not epoch%args.saveevery:
                self.save_models()
            
            if trainf:
                #critic.sched.step()
                pass
 
        print()

    def collect_dataset(self, path, size=2000, wait=10, datadir="./results/stuff/"):
        os.environ["MINERL_DATA_ROOT"] = "./data"
        #minerl.data.download("./data", experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0', num_workers=1)
        X = []
        Y = []
        cons = self.args.cons
        wait = self.args.wait
        delay = self.args.delay
        warmup = self.args.warmup
        chunksize = self.args.chunksize

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(10,2*wait, preload_buffer_size=1)):
            print("at batch", b_idx, end='\r')
            #vector = state['vector']

            # CONVERt COLOR
            pov = state['pov']
            if self.args.color == "HSV":
                pov = (255*rgb_to_hsv(pov/255)).astype(np.uint8)

            rewards = []
            approaches = []
            chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew>0, axis=1)) if pos>wait]
            print(np.max(rew, axis=1))
            print(chops)
            for chopidx,(rowidx, tidx) in enumerate(chops):
                rewards.extend([0]*chunksize)
                approaches.extend(pov[rowidx,warmup:warmup+chunksize])
                rewards.extend([1]*chunksize)
                approaches.extend(pov[rowidx,tidx-chunksize+1-delay:tidx+1-delay])
                
                effchsize = chunksize*2
                for chunkidx in range(effchsize):
                    if path.__contains__("test") and len(X)<500: # SAVE IMG
                        img = Image.fromarray(np.uint8(255*hsv_to_rgb(approaches[chopidx*effchsize+chunkidx]/255)))
                        #draw = ImageDraw.Draw(img)
                        #x, y = 0, 0
                        #draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]), fill= (255,255,255), font=self.font)
                        img.save(datadir+f"{b_idx}-{chopidx}-{chunkidx}-{'A' if rewards[chopidx*effchsize+chunkidx] else 'B'}.png")
                            
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_dataset_discounted(self, path, size=2000, wait=10, datadir="./results/stuff/"):
        os.environ["MINERL_DATA_ROOT"] = "./data"
        #minerl.data.download("./data", experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0')
        X = []
        Y = []
        cons = self.args.cons

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(10,cons)):
            print("at batch", b_idx, end='\r')
            #vector = state['vector']

            # CONVERt COLOR
            pov = state['pov']
            if self.args.color == "HSV":
                pov = (255*rgb_to_hsv(pov/255)).astype(np.uint8)

            gamma = self.args.gamma
            revgamma = self.args.revgamma
            stepsize = 2
            #chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>wait+stepsize*cons]
            #chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>wait and pos <cons]
            approaches = []
            rewards = []
            #rewimg = pov[rew==1][0]
            #print(rewimg.shape)
            #plt.imsave(f"./results/Critic/stuff/rewimg-{b_idx}.png", hsv_to_rgb(revimg/255))
            for ri,orow in enumerate(rew):
                chops = np.nonzero(orow)[0]
                #print(chops, row)
                if chops.size ==0:
                    continue
                end = np.max(chops)
                sequ = pov[ri,:end+1]
                orow = orow[:end+1]
                assert orow[-1]>0, "ERROR wrong chop detection"

                waitcount = wait
                rowrew = []
                selection = []
                addfak = 0
                revaddfak = 0
                for i in range(1, len(orow)+1):
                    waitcount -= 1

                    if orow[-i]>0: # RESET
                        fak = 1 #exponential
                        sub = 0 #subtraction
                        addfak += 1 #exponanential with add-reset
                        revfak = 1
                        revaddfak += 1
                        revhelper = 0.01
                        #fak = 0
                        waitcount = wait
                    if waitcount>0:
                        continue

                    selection.append(-i)
                    rewtuple = (fak, addfak, revfak, revaddfak, sub)
                    rowrew.append(rewtuple)

                    if path.__contains__("test") and len(X)<200: # SAVE IMG
                        img = Image.fromarray(np.uint8(255*hsv_to_rgb(sequ[-i]/255)))
                        draw = ImageDraw.Draw(img)
                        x, y = 0, 0
                        draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]), fill= (255,255,255), font=self.font)
                        img.save(datadir+f"{b_idx}-{ri}-{i}.png")


                    # DISCOUNT 
                    fak *= gamma
                    sub -= 1
                    addfak *= gamma
                    revfak = max(revfak-revhelper, 0)
                    revaddfak = max(revaddfak-revhelper, 0)
                    revhelper *= revgamma

                #print(row)
                rewards.extend(rowrew)
                approaches.extend(sequ[selection])
            
            #print(approaches)
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            #print(len(X))

            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)
        with gzip.GzipFile(path, 'wb') as fp:
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
    #parser.add_argument("-vizdataset", action="store_true")
    parser.add_argument("--gray", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--rewidx", type=int, default=3)
    parser.add_argument("--wait", type=int, default=100)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--revgamma", type=float, default=1.1)
    parser.add_argument("--datasize", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--cons", type=int, default=250)
    parser.add_argument("--color", type=str, default="HSV")
    parser.add_argument("--name", type=str, default="default")
    args = parser.parse_args()
    print(args)

    H = Handler(args)
    H.load_data()
    try:
        if args.load:
            H.load_models()
        if args.train:
            H.forward(mode="train")
            H.save_models()
        if args.test:
            if not args.train:
                H.load_models()
            H.forward(mode="test")

    except Exception:
        L.exception("Exception occured:")
        print("EXCEPTION")
    exit(0)
