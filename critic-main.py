import argparse
from matplotlib import pyplot as plt
import utils
import os
from nets import AutoEncoder, VAE, Critic, Unet
import numpy as np
import torch as T
import torch.nn as nn
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
        self.critic = Critic(end=[] if not args.sigmoid else [nn.Sigmoid()]).to(self.device)
        self.unet = Unet().to(self.device)
        self.models = dict()
        self.criticname = "critic"
        self.unetname = f"unet-l2_{args.L2}-l1_{args.L1}"
        self.models[self.criticname] = self.critic
        self.models[self.unetname] = self.unet
        if args.discounted:
            self.data_args = f"discount-{self.args.color}-ds{args.datasize}-cons{self.args.cons}-delay{self.args.delay}-gam{self.args.gamma}-revgam{self.args.revgamma}-chunk{self.args.chunksize}"
            self.data_path = f"data/discounted/tree-chop/{self.data_args}/"
        else:
            self.data_args = f"split-{self.args.color}-ds{args.datasize}-wait{args.wait}-delay{self.args.delay}-warmup{self.args.warmup}-chunk{self.args.chunksize}"
            self.data_path = f"data/split/tree-chop/{self.data_args}/"
        self.arg_path = self.data_args +"/"
        print("model path:", self.arg_path)
        self.result_path = f"./results/Critic/"+ args.name+ "-"+ self.arg_path
        print("viz path:", self.result_path)
        self.save_path = f"./saves/Critic/"+self.arg_path

        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 9)
        
        L.basicConfig(filename=f'./logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def load_data(self, batch_size = 64):
        wait = self.args.wait
        data_size = self.args.datasize
        test_size = 300
        data_path = self.data_path
        file_name = "data.pickle"
        data_collector = self.collect_discounted_dataset if self.args.discounted else self.collect_split_dataset

        print("loading data:", data_path)
        # TRAIN
        if not os.path.exists(data_path+file_name):
            print("train set...")
            os.makedirs(data_path, exist_ok=True)
            data_collector(data_path+file_name, size=data_size, datadir=data_path)
        # TEST
        if not os.path.exists(data_path+"test-"+file_name):
            print("collecting test set...")
            os.makedirs(data_path, exist_ok=True)
            data_collector(data_path+"test-"+file_name, size=test_size, datadir=data_path, test=10)


        # TRAIN data
        with gzip.open(data_path+file_name, "rb") as fp:
           self.X, self.Y = pickle.load(fp)
        self.dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.X), T.from_numpy(self.Y)), batch_size=batch_size, shuffle=True)
        print(f"loaded train set with {len(self.X)}")
        # TEST data
        with gzip.open(data_path+"test-"+file_name, "rb") as fp:
            self.XX, self.YY = pickle.load(fp)
        self.testdataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.XX), T.from_numpy(self.YY)), batch_size=300, shuffle=False)
        print(f"loaded test set with {len(self.XX)}")

    def load_models(self, modelnames=[]):
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            print("loading:", self.save_path+f'{model}.pt')
            self.models[model].load_state_dict(T.load(self.save_path+f'{model}.pt', map_location=T.device(self.device)))

    def save_models(self, modelnames=[]):
        os.makedirs(self.save_path, exist_ok=True)
        save_path = self.save_path
        if not modelnames:
            modelnames = self.models.keys()
        for model in modelnames:
            print("saving:", save_path+f'{model}.pt')
            T.save(self.models[model].state_dict(), save_path+f'/{model}.pt')

    def critic_pipe(self, mode="train", test=0):
        args = self.args
        testf = mode=="test"
        trainf = mode=="train"
        loader = self.dataloader if trainf else self.testdataloader

        # Setup save path and Logger
        result_path = self.result_path+"critic/"+mode+"/" 
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
                if args.discounted:
                    Y = Y[:,args.rewidx].float().to(self.device)
                else:
                    Y = Y.float().to(self.device)
                #print(X.shape, Y.shape, Y)

                pred = critic(XP).squeeze()
                YY = T.sigmoid(pred)
                if trainf:
                    if args.discounted:
                        loss = F.mse_loss(pred, Y)
                    else:
                        loss = F.binary_cross_entropy_with_logits(pred, Y)
                    print(f"critic e{epoch} b{b_idx}", loss.item(), end="\r")
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                #log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx%100) or testf: # VISUALIZE
                    if False:
                        order1 = YY.argsort(descending=True)
                        order2 = Y.argsort(descending=True)
                    if trainf:
                        L.info(f"critic e{epoch} b{b_idx} loss: {loss.item()}")
                    order1 = np.arange(Y.shape[0])
                    order2 = np.arange(Y.shape[0])
                    viz = hsv_to_rgb(X[order1].numpy()/255) if self.args.color == "HSV" else X[order].numpy()/255
                    viz = np.concatenate(viz, axis=1)
                    viz2 = hsv_to_rgb(X[order2].numpy()/255) if self.args.color == "HSV" else X[order].numpy()/255
                    viz2 = np.concatenate(viz2, axis=1)

                    viz = np.concatenate((viz,viz2), axis=0)
                    img = Image.fromarray(np.uint8(255*viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(YY[order1].tolist()):
                        x, y = int(i*img.width/len(YY)), 1
                        draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)
                    for i, value in enumerate(Y[order2].tolist()):
                        x, y = int(i*img.width/len(Y)), int(1+img.height/2)
                        draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)

                    #plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path+f"e{epoch}_b{b_idx}.png")
            
            if epoch and not epoch%args.saveevery:
                self.save_models(modelnames=[self.criticname])
            
            if trainf:
                #critic.sched.step()
                pass
 
        print()

    def dream(self):
        args = self.args
        # Setup save path and Logger
        result_path = self.result_path+f"dreamsteps{args.dreamsteps}"+"/" 
        os.makedirs(result_path, exist_ok=True)
        loader = self.testdataloader
        critic = self.critic
        dreamsteps = args.dreamsteps

        # Epoch and Batch Loops
        for b_idx, (X,Y) in enumerate(loader):
            # FORWARD PASS---------------------------

            XP = (X.permute(0,3,1,2).float().to(self.device)).requires_grad_()
            
            opti = T.optim.Adam([XP], lr=0.1)
            if args.discounted:
                Y = Y[:,args.rewidx].float().to(self.device)
            else:
                Y = Y.float().to(self.device)

            pred = critic(XP).squeeze()
            Yorig = T.sigmoid(pred)
            pre = XP.data.detach().clone()
            for upidx in range(dreamsteps):
                if args.discounted:
                    loss = F.mse_loss(pred, T.zeros_like(pred))
                else:
                    loss = F.binary_cross_entropy_with_logits(pred, T.zeros_like(pred))
                #print(f"b{b_idx} up-step {upidx}", loss.item(), end="\r")

                #opti.zero_grad()
                if XP.grad is not None:
                    XP.grad *= 0

                loss.backward()
                #print("grad", XP.grad[0])
                XP.data = XP.data - 10*XP.grad.data
                #print((XP.data-pre).mean() - XP.grad.data.mean())
                #print(XP.requires_grad)
                #opti.step()
                pred = critic(XP).squeeze()
                #print(XP.requires_grad)
                Yfinal = T.sigmoid(pred)
                #print("grad", XP.grad[0])
            print((XP-pre).data.max())
            #log_file.write(log_msg+"\n")

            # VIZ -----------------------------------
            XP = XP.detach().permute(0,2,3,1).numpy()/255
            pre = pre.detach().permute(0,2,3,1).numpy()/255
            viz3 = np.zeros_like(XP.numpy())
            viz3[:,:,2] = np.abs(np.mean((XP-pre).numpy(), axis=-1))
            viz = hsv_to_rgb(XP) if self.args.color == "HSV" else XP
            viz = np.concatenate(viz, axis=1)
            viz2 = hsv_to_rgb(X.numpy()/255) if self.args.color == "HSV" else X.numpy()/255
            viz2 = np.concatenate(viz2, axis=1)
            

            viz = np.concatenate((viz,viz2,viz3), axis=0)
            img = Image.fromarray(np.uint8(255*viz))
            draw = ImageDraw.Draw(img)
            for i, value in enumerate(Yfinal.tolist()):
                x, y = int(i*img.width/len(Yfinal)), 1
                draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)
            for i, value in enumerate(Yorig.tolist()):
                x, y = int(i*img.width/len(Yorig)), int(1+img.height/2)
                draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)

            #plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
            img.save(result_path+f"b{b_idx}.png")

        print()

    def segment(self, mode="train", test=0):
        args = self.args
        testf = mode=="test"
        trainf = mode=="train"
        loader = self.dataloader if trainf else self.testdataloader
        critic = self.critic
        unet = self.unet
        opti = T.optim.Adam(unet.parameters(recurse=True))
        if args.live:
            critic_opti = T.optim.Adam(critic.parameters(recurse=True))

        # Setup save path and Logger
        result_path = self.result_path+"segment/"+mode+"/" 
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{self.args}\n\n")
        if trainf:
            log = []
        loss_string = ""
        

        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            for b_idx, (X,Y) in enumerate(loader):
                # FORWARD PASS---------------------------
                XP = X.permute(0,3,1,2).float().to(self.device)
                if args.discounted:
                    Y = Y[:,args.rewidx].float().to(self.device)
                else:
                    Y = Y.float().to(self.device)

                # FILTER in A and B
                if trainf:
                    if args.live:
                        pred = T.sigmoid(critic(XP).squeeze())
                        critic_loss = F.binary_cross_entropy_with_logits(pred, Y)
                        critic_opti.zero_grad()
                        critic_loss.backward()
                        critic_opti.step()
                        loss_string = f"live-critic {critic_loss.item()   }   " + loss_string
                    else:
                        with T.no_grad():
                            pred = T.sigmoid(critic(XP).squeeze())
                    #mask = pred>args.threshold
                    negmask = pred<(1-args.threshold)
                    if not negmask.sum():
                        print(negmask.sum())
                        continue
                    #print(negmask.shape)
                    #print(np.nonzero(negmask))
                    #print(np.nonzero(negmask.numpy())[0])
                    negatives = np.random.choice((np.nonzero(negmask.cpu().numpy()))[0], len(X))
                    A = XP
                    B = XP[negatives]
                    Z = unet(A)
                    merged = A*(1-Z) + Z*B
                    mergevalue = critic(merged).squeeze()
                    valueloss = F.binary_cross_entropy_with_logits(mergevalue, T.zeros_like(mergevalue))
                    loss = valueloss
                    if args.L1:
                        normloss = args.L1*F.l1_loss(Z, T.zeros_like(Z))
                        loss = loss + normloss
                        loss_string = f"L1: {normloss.item()   }   " + loss_string
                    elif args.L2:
                        normloss = args.L2*F.mse_loss(Z, T.zeros_like(Z))
                        loss = loss + normloss
                        loss_string = f"L2: {normloss.item()   }   " + loss_string
                    if args.distnorm:
                        mask = Z.cpu().detach()
                        w = X.shape[1]
                        b = X.shape[0]
                        xs = T.arange(w).repeat((b, 1, w, 1)).float()/w
                        ys = T.arange(w).repeat((b, 1, w, 1)).transpose(2, 3).float()/w
                        #print(xs[0], xs[1], xs.shape, ys.shape, mask.shape)
                        xvote = (xs*mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b,1,1,1)
                        yvote = (ys*mask).flatten(start_dim=-2).mean(dim=-1).squeeze().view(b,1,1,1)
                        print(xs.shape, xvote.shape)
                        xs -= xvote # X Distance
                        ys -= yvote # Y Distance
                        dist = (xs.pow(2)+xs.pow(2)).pow(0.5)
                        target = mask-dist
                        distloss = F.mse_loss(Z, target.to(self.device))
                        loss = loss + distloss
                        loss_string = f"dist-norm: {distloss.item()   }" + loss_string

                    
                    mergevalue = T.sigmoid(mergevalue)
                    loss_string = f"e{epoch} b{b_idx}  value-loss: {loss.item()}   " + loss_string 
                    print(loss_string)
                    log.append((valueloss.item(), normloss.item()))
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                #log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx%100): # VISUALIZE
                    A = A.cpu().detach().permute(0,2,3,1)
                    B = B.cpu().detach().permute(0,2,3,1)
                    Z = Z.cpu().detach().permute(0,2,3,1)
                    #print("SHAPE", Z.shape)
                    merged = merged.cpu().detach().permute(0,2,3,1)
                    viz1 = hsv_to_rgb(A.numpy()/255) if self.args.color == "HSV" else A.numpy()/255
                    viz1 = np.concatenate(viz1, axis=1)
                    viz2 = hsv_to_rgb(B.numpy()/255) if self.args.color == "HSV" else B.numpy()/255
                    viz2 = np.concatenate(viz2, axis=1)
                    viz3 = hsv_to_rgb(merged.numpy()/255) if self.args.color == "HSV" else merged.numpy()/255
                    viz3 = np.concatenate(viz3, axis=1)
                    viz4 = T.cat((Z,Z,Z), dim=-1).cpu().numpy()
                    viz4 = np.concatenate(viz4, axis=1)

                    viz = np.concatenate((viz1,viz2,viz3,viz4), axis=0)
                    img = Image.fromarray(np.uint8(255*viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(pred.tolist()):
                        x, y = int(i*img.width/len(pred)), 1
                        draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)
                    for i, value in enumerate(pred[negatives].tolist()):
                        x, y = int(i*img.width/len(pred[negatives])), int(1+img.height/3)
                        draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)
                    for i, value in enumerate(mergevalue.tolist()):
                        x, y = int(i*img.width/len(mergevalue)), int(1+2*img.height/3)
                        draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)

                    #plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path+f"e{epoch}_b{b_idx}.png")
            
                if testf: # VISUALIZE
                    viz1 = hsv_to_rgb(X.numpy()/255) if self.args.color == "HSV" else A.numpy()/255
                    viz1 = np.concatenate(viz1, axis=1)
                    Z = unet(XP)
                    Z = Z.detach().permute(0,2,3,1)
                    seg = X.float()/255
                    seg[:,:,:,2] = Z.squeeze()
                    viz2 = hsv_to_rgb(seg.numpy()) if self.args.color == "HSV" else seg.numpy()
                    viz2 = np.concatenate(viz2, axis=1)
                    viz4 = T.cat((Z,Z,Z), dim=-1).cpu().numpy()
                    viz4 = np.concatenate(viz4, axis=1)

                    viz = np.concatenate((viz1,viz2,viz4), axis=0)
                    img = Image.fromarray(np.uint8(255*viz))
                    draw = ImageDraw.Draw(img)
                    YY = T.sigmoid(critic(XP)).squeeze()
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i*img.width/len(YY)), 1
                        draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)

                    #plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path+f"e{epoch}_b{b_idx}.png")

            if epoch and not epoch%args.saveevery:
                self.save_models(modelnames=[self.unetname])
            
            if trainf:
                #critic.sched.step()
                pass
        if trainf:
            log = np.array(log)
            end = len(log)//10
            log1 = log[:10*end,0]
            log1 = log1.reshape((-1,10))
            log1 = log1.mean(axis=-1)
            plt.plot(log1, label="value loss")
            log2 = log[:10*end,1]
            log2 = log2.reshape((-1,10))
            log2 = log2.mean(axis=-1)
            plt.plot(log2, label="L2 norm")
            plt.legend()
            plt.savefig(result_path+f"loss.png")
        print()        

    def collect_split_dataset(self, path, size=2000, wait=10, test=0, datadir="./results/stuff/"):
        os.makedirs(datadir+"samples/", exist_ok=True)
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
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(test or 10, 2*wait if not test else size, preload_buffer_size=1)):
            print("at batch", b_idx, end='\r')
            #vector = state['vector']

            # CONVERt COLOR
            pov = state['pov']
            if self.args.color == "HSV":
                pov = (255*rgb_to_hsv(pov/255)).astype(np.uint8)

            rewards = []
            approaches = []
            if test:
                for rowidx in range(len(rew)):
                    rewards.extend(rew[rowidx])
                    approaches.extend(pov[rowidx])
            else:
                chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew==1, axis=1)) if pos>wait]
                #print(np.max(rew, axis=1))
                #print(chops)
                for chopidx,(rowidx, tidx) in enumerate(chops):
                    rewards.extend([0]*chunksize)
                    approaches.extend(pov[rowidx,warmup:warmup+chunksize])
                    rewards.extend([1]*chunksize)
                    approaches.extend(pov[rowidx,tidx-chunksize+1-delay:tidx+1-delay])
                    
                    if len(X)<500: # SAVE IMG
                        effchsize = chunksize*2
                        for chunkidx in range(effchsize):
                            img = Image.fromarray(np.uint8(255*hsv_to_rgb(approaches[chopidx*effchsize+chunkidx]/255)))
                            #draw = ImageDraw.Draw(img)
                            #x, y = 0, 0
                            #draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]), fill= (255,255,255), font=self.font)
                            img.save(datadir+"samples/"+f"{b_idx}-{chopidx}-{chunkidx}-{'A' if rewards[chopidx*effchsize+chunkidx] else 'B'}.png")
                                
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            if test:
                break

            print(len(X))
            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break


        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_discounted_dataset(self, path, size=2000, datadir="./results/stuff/", test=0):
        os.makedirs(datadir+"samples/", exist_ok=True)
        os.environ["MINERL_DATA_ROOT"] = "./data"
        #minerl.data.download("./data", experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0')
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        if test:
            testsize= size
            size= size*test

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(10,cons if not test else testsize)):
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
                end = np.max(chops) if not test else testsize-1
                sequ = pov[ri,:end+1]
                orow = orow[:end+1]
                assert test or orow[-1]>0, "ERROR wrong chop detection"

                delaycount = delay
                rowrew = []
                selection = []
                addfak = 0
                revaddfak = 0
                if test:
                    fak = 0
                    sub = 0
                    revfak = 0
                    revhelper = 0.01

                for i in range(1, len(orow)+1):
                    delaycount -= 1

                    if orow[-i]>0: # RESET
                        fak = 1 #exponential
                        sub = 0 #subtraction
                        addfak += 1 #exponanential with add-reset
                        revfak = 1
                        revaddfak += 1
                        revhelper = 0.01
                        #fak = 0
                        delaycount = delay
                    if delaycount>0:
                        continue

                    selection.append(-i)
                    rewtuple = (fak, addfak, revfak, revaddfak, sub)
                    rowrew.append(rewtuple)

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

            if len(X)<300: # SAVE IMG
                for fi, frame in enumerate(approaches):
                    img = Image.fromarray(np.uint8(255*hsv_to_rgb(frame/255)))
                    draw = ImageDraw.Draw(img)
                    #print(rewards)
                    rewtuple = rewards[fi]
                    x, y = 0, 0
                    draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]), fill= (255,255,255), font=self.font)
                    img.save(datadir+"samples/"+f"{b_idx}-{fi}.png")

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
    parser.add_argument("-unet", action="store_true")
    parser.add_argument("-live", action="store_true")
    parser.add_argument("-critic", action="store_true")
    parser.add_argument("-distnorm", action="store_true")
    parser.add_argument("-uload", action="store_true")
    parser.add_argument("-cload", action="store_true")
    parser.add_argument("-dream", action="store_true")
    parser.add_argument("-discounted", action="store_true")
    parser.add_argument("-sigmoid", action="store_true")
    #parser.add_argument("-vizdataset", action="store_true")
    parser.add_argument("--gray", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dreamsteps", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--L2", type=float, default=0.0)
    parser.add_argument("--L1", type=float, default=0.0)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--rewidx", type=int, default=3)
    parser.add_argument("--wait", type=int, default=120)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--revgamma", type=float, default=1.1)
    parser.add_argument("--datasize", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--cons", type=int, default=250)
    parser.add_argument("--color", type=str, default="HSV")
    parser.add_argument("--name", type=str, default="default")
    args = parser.parse_args()
    print(args)

    H = Handler(args)
    H.load_data()
    try:
        if args.cload:
            H.load_models(modelnames=[H.criticname])
        if args.uload:
            H.load_models(modelnames=[H.unetname])
        if args.train:
            if args.critic:
                H.critic_pipe(mode="train")
                H.save_models(modelnames=[H.criticname])
            if args.unet:
                H.load_models(modelnames=[H.criticname])
                H.segment(mode="train")
                H.save_models(modelnames=[H.unetname])
        if args.test:
            if not args.train:
                H.load_models()
            if args.critic:
                H.critic_pipe(mode="test")
            if args.unet:
                H.segment(mode="test")
        if args.dreamsteps:
            if not args.train:
                H.load_models()
            H.dream()


    except Exception as e:
        L.exception("Exception occured:"+ str(e))
        print("EXCEPTION")
        print(e)
    exit(0)
