import argparse
from matplotlib import pyplot as plt
import os
from nets import AutoEncoder, VAE, Critic, Unet, ResNetCritic, GroundedUnet
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import minerl
import utils
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from matplotlib import colors
from matplotlib import cm
import io
import cv2
from sklearn.cluster import KMeans
import logging as L
import gzip
import math
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torchvision as tv
from PatchEmbedder import PatchEmbedder
from sklearn.mixture import GaussianMixture as GMM
import copy

class Handler():
    def __init__(self, args):
        #os.environ["MINERL_DATA_ROOT"] = "data"
        #self.data = minerl.data.make('MineRLTreechopVectorObf-v0')
        self.args = args
        self.device = "cuda" if T.cuda.is_available() else "cpu"
        #self.device = "cpu"
        print("device:", self.device)
        if args.resnet:
            self.critic = ResNetCritic().to(self.device)
            args.color = "RGB"
        else:
            self.critic = Critic(end=[] if not args.sigmoid else [nn.Sigmoid()], colorchs= args.clustercritic+3 if args.clustercritic else 3).to(self.device)
        if args.grounded:
            self.unet = GroundedUnet().to(self.device)
        else:
            self.unet = Unet().to(self.device)
        self.models = dict()
        self.criticname = "critic"+ ("+5" if args.clustercritic else "")
        if args.discounted:
            self.data_args = f"discount-{self.args.color}-ds{args.datasize}-cons{self.args.cons}-delay{self.args.delay}-gam{self.args.gamma}-revgam{self.args.revgamma}-chunk{self.args.chunksize}"
            if args.integrated:
                self.data_path = f"./isy_minerl/segm/data/discounted/tree-chop/{self.data_args}/"
            else:
                self.data_path = f"./data/discounted/tree-chop/{self.data_args}/"
        else:
            self.data_args = f"split-{self.args.color}-ds{args.datasize}-wait{args.wait}-delay{self.args.delay}-warmup{self.args.warmup}-chunk{self.args.chunksize}"
            if args.integrated:
                self.data_path = f"./isy_minerl/segm/data/split/tree-chop/{self.data_args}/"
            else:
                self.data_path = f"./data/split/tree-chop/{self.data_args}/"
        self.arg_path = f"{'grounded-' if args.grounded else ''}{'resnet-' if args.resnet else ''}{'blur'+str(args.blur)+'-' if args.blur else ''}{'L1_'+str(args.L1)+'-'}"+self.data_args +"/"
        print("model path:", self.arg_path)
        if args.integrated:
            self.result_path = f"./isy_minerl/segm/results/Critic/"+ args.name+ "-"+ self.arg_path
        else:
            self.result_path = f"./results/Critic/"+ args.name+ "-"+ self.arg_path
        print("viz path:", self.result_path)

        if args.final or args.integrated:
            self.unetname = f"unet"
            self.embed_data_path = f"./train/"
            self.embed_data_args = "embed-data"
            self.save_path = f"./train/"
            self.font = ImageFont.truetype("./isy_minerl/segm/etc/Ubuntu-R.ttf", 9)
        else:
            self.embed_data_path = f"saves/patchembed/"
            self.embed_data_args = f"cl{args.embed_cluster}-dim{args.embed_dim}-ds{args.embed_train_samples}-" \
                                   + f"dl{args.delay}-th{args.embed_pos_threshold}-pw{args.embed_patch_width}" \
                                    + f"{'-hue' if args.hue else '-hs'}"
            self.unetname = f"unet-l2_{args.L2}-l1_{args.L1}"
            self.save_path = f"./saves/Critic/"+self.arg_path
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 8)

        self.models[self.criticname] = self.critic
        self.models[self.unetname] = self.unet


        L.basicConfig(filename=f'./logs/{args.name}.log', format='%(asctime)s %(levelname)s %(name)s %(message)s', level=L.INFO)

    def load_data(self, batch_size = 64):
        args = self.args
        wait = self.args.wait
        data_size = self.args.datasize
        test_size = args.testsize
        data_path = self.data_path
        file_name = "data.pickle"
        data_collector = self.collect_discounted_dataset if self.args.discounted else self.collect_split_dataset

        if args.blur:
            #blur = nn.AvgPool2d(args.blur, 1, 1)
            if args.blur==3:
                blurkernel = T.tensor([[[[1,2,1],[2,4,2], [1,2,1]]]*1]*3).float()/16
                blur = lambda x: F.conv2d(x, blurkernel, padding=1, groups=3)
            if args.blur ==5:
                blurkernel = T.tensor([[[[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]]]*1]*3).float()/256
                blur = lambda x: F.conv2d(x, blurkernel, padding=2, groups=3)

        print("loading data:", data_path)
        # TRAIN
        if not os.path.exists(data_path+file_name):
            print("train set...")
            os.makedirs(data_path, exist_ok=True)
            data_collector(data_path+file_name, size=data_size, datadir=data_path)
        # TEST
        testfilename = data_path+"test-"+file_name
        if not os.path.exists(testfilename):
            print("collecting test set...")
            os.makedirs(data_path, exist_ok=True)
            data_collector(testfilename, size=test_size, datadir=data_path, test=10)

        if args.clippify:
            utils.data_set_to_vid(testfilename, HSV=args.color=="HSV")

        # TRAIN data
        with gzip.open(data_path+file_name, "rb") as fp:
           self.X, self.Y = pickle.load(fp)
           if args.blur:
               self.X = np.swapaxes(self.X, 1,-1)
               self.X = blur(T.from_numpy(self.X).float()).numpy().astype(np.uint8)
               self.X = np.swapaxes(self.X, 1,-1)
               print(self.X.shape)

        self.dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.X), T.from_numpy(self.Y), T.arange(self.X.shape[0], dtype=T.uint8)), batch_size=batch_size, shuffle=True)
        self.trainsize = self.X.shape[0]
        print(f"loaded train set with {len(self.X)}")
        # TEST data
        with gzip.open(testfilename, "rb") as fp:
            self.XX, self.YY = pickle.load(fp)
            if args.blur:
                self.XX = np.swapaxes(self.XX, 1,-1)
                self.XX = blur(T.from_numpy(self.XX).float()).numpy().astype(np.uint8)
                self.XX = np.swapaxes(self.XX, 1,-1)
                print(self.XX.shape)
        self.testdataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(T.from_numpy(self.XX), T.from_numpy(self.YY), T.arange(self.XX.shape[0], dtype=T.uint8)), batch_size=300, shuffle=False)
        self.testsize = self.XX.shape[0]
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

        if args.clustercritic:
            with gzip.open(self.data_path+f"{mode}-{str(self.args.clustercritic)}-cluster", 'rb') as fp:
                channels = pickle.load(fp)


        # Setup save path and Logger
        result_path = self.result_path+"critic/"+mode+"/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        critic = self.critic
        if args.resnet:
            opti = T.optim.Adam(critic.head.parameters())
        else:
            opti = T.optim.Adam(critic.parameters())
        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            for b_idx, (X,Y,I) in enumerate(loader):
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

                if not args.clustercritic:
                    pred = critic(XP).squeeze()
                else:
                    CHS = T.from_numpy(channels[I]).float().to(self.device)
                    XPE = T.cat((XP,CHS), dim=1)
                    pred = critic(XPE).squeeze()
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
                    vizs = []

                    if False:
                        order1 = YY.argsort(descending=True)
                        order2 = Y.argsort(descending=True)
                    if trainf:
                        L.info(f"critic e{epoch} b{b_idx} loss: {loss.item()}")
                    viz = hsv_to_rgb(X.numpy()/255) if self.args.color == "HSV" else X.numpy()/255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    for chi in range(args.clustercritic):
                        chmap = CHS[chi].cpu().numpy()
                        print(chmap.shape)
                        segmap = np.stack((chmap, chmap, chmap), axis = -1)
                        viz = np.concatenate(segmap, axis=1)
                        vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
                    img = Image.fromarray(np.uint8(255*viz))
                    draw = ImageDraw.Draw(img)
                    for i, value in enumerate(YY.tolist()):
                        x, y = int(i*img.width/len(YY)), 1
                        draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)
                    for i, value in enumerate(Y.tolist()):
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
        for b_idx, (X,Y,I) in enumerate(loader):
            # FORWARD PASS---------------------------

            XP = (X.permute(0,3,1,2).float().to(self.device)).requires_grad_()

            opti = T.optim.Adam([XP], lr=0.1)
            if args.discounted:
                Y = Y[:,args.rewidx].float().to(self.device)
            else:
                Y = Y.float().to(self.device)

            pred = critic(XP).squeeze()
            original_value = T.sigmoid(pred)
            original_img = XP.data.detach().clone()
            for upidx in range(dreamsteps):
                if args.discounted:
                    loss = F.mse_loss(pred, T.zeros_like(pred))
                else:
                    loss = F.binary_cross_entropy_with_logits(pred, T.zeros_like(pred))
                #print(f"b{b_idx} up-step {upidx}", loss.item(), end="\r")
                critic.zero_grad()
                loss.backward()
                #print("grad", XP.grad[0])
                avg_grad = np.abs(XP.grad.data.cpu().numpy()).mean()
                norm_lr = 0.01/avg_grad
                XP.data += norm_lr*avg_grad
                XP = XP.clamp(0,255)
                pred = critic(XP).squeeze()
                final_value = T.sigmoid(pred)
                XP.grad.data.zero_()
                #print(XP.requires_grad)
                #print("grad", XP.grad[0])
            print((XP-original_img).data.max())
            #log_file.write(log_msg+"\n")

            # VIZ -----------------------------------
            XP = XP.detach().permute(0,2,3,1).numpy()/255
            pre = pre.detach().permute(0,2,3,1).numpy()/255
            viz3 = np.zeros_like(XP.numpy())
            viz3[:,:,2] = np.abs(np.mean((XP-original_img).numpy(), axis=-1))
            viz = hsv_to_rgb(XP) if self.args.color == "HSV" else XP
            viz = np.concatenate(viz, axis=1)
            viz2 = hsv_to_rgb(X.numpy()/255) if self.args.color == "HSV" else X.numpy()/255
            viz2 = np.concatenate(viz2, axis=1)


            viz = np.concatenate((viz,viz2,viz3), axis=0)
            img = Image.fromarray(np.uint8(255*viz))
            draw = ImageDraw.Draw(img)
            for i, value in enumerate(final_value.tolist()):
                x, y = int(i*img.width/len(Yfinal)), 1
                draw.text((x, y), str(round(value,3)), fill= (255,255,255), font=self.font)
            for i, value in enumerate(original_value.tolist()):
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
            if args.resnet:
                critic_opti = T.optim.Adam(critic.head.parameters(recurse=True), lr=1e-4)
            else:
                critic_opti = T.optim.Adam(critic.parameters(recurse=True), lr=1e-4)
        if args.clustercritic:
            with gzip.open(self.data_path+f"{str(self.args.clustercritic)}-cluster", 'rb') as fp:
                channels = pickle.load(fp)

        # Setup save path and Logger
        result_path = self.result_path+"segment/"+mode+"/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{self.args}\n\n")
        if trainf:
            log = []


        # Epoch and Batch Loops
        for epoch in range(int(testf) or self.args.epochs):
            for b_idx, (X,Y,I) in enumerate(loader):
                loss_string = ""
                # FORWARD PASS---------------------------
                XP = X.permute(0,3,1,2).float().to(self.device)
                if args.discounted:
                    Y = Y[:,args.rewidx].float().to(self.device)
                else:
                    Y = Y.float().to(self.device)

                # FILTER in A and B
                if trainf:
                    if not args.clustercritic:
                        pred = critic(XP).squeeze()
                    else:
                        CHS = T.from_numpy(channels[I]).float().to(self.device)
                        XPE = T.cat((XP,CHS), dim=1)
                        pred = critic(XPE).squeeze()
                    pred = T.sigmoid(pred)
                    if args.live:
                        critic_loss = F.binary_cross_entropy_with_logits(pred, Y)
                        critic_opti.zero_grad()
                        critic_loss.backward()
                        critic_opti.step()
                        loss_string = f"live-critic {critic_loss.item()   }   " + loss_string

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
                    if not args.clustercritic:
                        mergevalue = critic(merged).squeeze()
                    else:
                        mergechs = CHS*(1-Z) + Z*CHS[negatives]
                        mergecombined = T.cat((merged,mergechs), dim=1)
                        mergevalue = critic(mergecombined).squeeze()
                    valueloss = F.binary_cross_entropy_with_logits(mergevalue, T.zeros_like(mergevalue))
                    loss = valueloss
                    if args.L1:
                        if args.staticL1:
                            valuefak = 1
                        else:
                            valuefak = 1-pred.detach().view(-1,1,1,1)
                        normloss = args.L1 * F.l1_loss(valuefak*Z, T.zeros_like(Z))
                        # normloss = args.L1 * (valuefak*Z).mean()
                        loss = loss + normloss
                        loss_string = f"L1: {normloss.item()   }   " + loss_string
                    elif args.L2:
                        if args.staticL1:
                            valuefak = 1
                        else:
                            valuefak = 1-pred.detach().view(-1,1,1,1)
                        normloss = args.L2*F.mse_loss(valuefak*Z, T.zeros_like(Z))
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
                        #print(xs.shape, xvote.shape)
                        xs -= xvote # X Distance
                        ys -= yvote # Y Distance
                        dist = (xs.pow(2)+xs.pow(2)).pow(0.5)
                        target = mask-dist
                        target[target<0] = 0
                        distloss = 5*F.mse_loss(Z, target.to(self.device))
                        loss = loss + distloss
                        loss_string = f"dist-norm: {distloss.item()   }   " + loss_string


                    mergevalue = T.sigmoid(mergevalue)
                    loss_string = f"e{epoch} b{b_idx}  value-loss: {loss.item()}   " + loss_string
                    print((loss_string))
                    log.append((valueloss.item(), normloss.item() if args.L1 or args.L2 else 0))
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                #log_file.write(log_msg+"\n")

                # VIZ -----------------------------------
                if (trainf and not b_idx%100): # VISUALIZE
                    vizs = []
                    A = A.cpu().detach().permute(0,2,3,1)
                    B = B.cpu().detach().permute(0,2,3,1)
                    Z = Z.cpu().detach().permute(0,2,3,1)
                    #print("SHAPE", Z.shape)
                    merged = merged.cpu().detach().permute(0,2,3,1)
                    viz = hsv_to_rgb(A.numpy()/255) if self.args.color == "HSV" else A.numpy()/255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = hsv_to_rgb(B.numpy()/255) if self.args.color == "HSV" else B.numpy()/255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = hsv_to_rgb(merged.numpy()/255) if self.args.color == "HSV" else merged.numpy()/255
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)
                    viz = T.cat((Z,Z,Z), dim=-1).cpu().numpy()
                    viz = np.concatenate(viz, axis=1)
                    vizs.append(viz)

                    viz = np.concatenate(vizs, axis=0)
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
                    seg[:,:,:,1] = Z.squeeze()
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

    def cluster(self, mode="train", test=0):
        args = self.args
        testf = mode=="test"
        trainf = mode=="train"
        loader = self.dataloader if trainf else self.testdataloader
        batchsize = loader.batch_size
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 14)
        size = self.trainsize if trainf else self.testsize
        assert not(args.clustersave and args.savekmeans)
        if args.clustersave:
            cdict = dict([(k,np.zeros((size, int(k), 64, 64))) for k in args.cluster.split(',')])

        treemask = np.zeros((64,64), dtype=np.uint8)
        treemask[21:42,25:39] = 1

        # Setup save path and Logger
        result_path = self.result_path+"cluster/"+mode+"/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        if args.savekmeans:
            sel = np.random.choice(np.arange(len(self.X)), 256)
            loader = [(T.from_numpy(self.X[sel]),self.Y[sel],0)]
            batchsize = 256

        chs = [0,1]
        # Epoch and Batch Loops
        for epoch in range(1):
            for b_idx, (X,Y,I) in enumerate(loader):
                if args.clustersave:
                    print(f"clustering dataset: {b_idx}/{math.ceil(args.datasize/batchsize)}")
                else:
                    print("generating kmeans...")
                if args.color == "RGB":
                    X = rgb_to_hsv(X)

                pixels = X.view(-1,3)
                test = pixels.view(X.shape)
                assert (X==test).all()
                pixels = pixels[:,chs].float()/255
                pixels[:,1] *= 0.1

                plt.ylim(0, 1)
                plt.hist2d(pixels[:,0].numpy(), pixels[:,1].numpy(), 100)
                plt.savefig(result_path+f"e{epoch}_b{b_idx}_scatter.png")

                vizs = []
                viz = hsv_to_rgb(X.numpy()/255) if self.args.color == "HSV" else X.numpy()/255
                treemaskviz = viz.copy()
                treemaskviz[:,treemask==0] *= 0.5
                viz = np.concatenate(viz, axis=1)
                treemaskviz = np.concatenate(treemaskviz, axis=1)
                vizs.append(viz)
                vizs.append(treemaskviz)
                text = []
                for nc in [int(number) for number in args.cluster.split(',')]:

                    clusters =  KMeans(n_clusters=nc).fit(pixels)
                    labels = clusters.labels_.reshape(X.shape[:-1])

                    clusterlayers = []
                    clustervalues = []
                    for clidx in range(nc):
                        row = X.numpy()
                        labelselect = labels==clidx
                        clusterlayers.append(labelselect)


                        # determine values
                        if not args.savekmeans:
                            Ylabel = Y.numpy()
                        else:
                            Ylabel = Y
                        tmpsel = Ylabel == 1
                        tmptm = np.tile(treemask, (int(np.sum(tmpsel)),1,1))
                        #print(tmptm.shape, labelselect[tmpsel].shape)
                        clustervalues.append(np.sum(labelselect[tmpsel]*tmptm)/np.sum(labelselect[tmpsel]))
                        #print(clidx, "clustervalue:", clustervalues[-1])

                        if False: #Spatial Clustering
                            for pi in range(len(row)):
                                w = X.shape[1]
                                xs = T.arange(w).repeat((w, 1)).float()/w
                                ys = T.arange(w).repeat((w, 1)).transpose(0, 1).float()/w
                                positions = T.stack((xs,ys), dim=-1)
                                flat_pos = positions.view(-1,2)
                                sublabels = np.zeros((w,w))

                                for li in range(nc):
                                    square_selection = labels[pi]==li
                                    flat_selection = square_selection.reshape(-1)
                                    #print(positions, positions.shape)
                                    sub = KMeans(n_clusters=2).fit(flat_pos[flat_selection])
                                    flat_sub_labels = sub.labels_
                                    sublabels[square_selection] = flat_sub_labels

                            row[pi,:,:,0] = 255 - 255 * ((labels[pi] / (nc*2)) * (1+sublabels))

                        row[:,:,:,1] = 255 * ((labelselect))

                        if b_idx <10 and args.viz: #"VISUALIZE"
                            viz = hsv_to_rgb(row/255) if self.args.color == "HSV" else row/255
                            viz[:,treemask==0] *= 0.5
                            viz = np.concatenate(viz, axis=1)
                            vizs.append(viz)
                            segmap = np.stack((labelselect, labelselect, labelselect), axis = -1)
                            viz = np.concatenate(segmap, axis=1)
                            vizs.append(viz)
                            text.append(f"{nc} {clidx}\n{clustervalues[clidx]}")

                    # SAVE KMEANS
                    targetcluster = np.argmax(clustervalues)
                    print("Target Cluster:", targetcluster)
                    if args.savekmeans:
                        with open(self.save_path+f'kmeans.p', 'wb') as fp:
                            pickle.dump((clusters, targetcluster), fp)

                    clusterlayers = np.transpose(np.array(clusterlayers, dtype=np.uint8), axes=(1,0,2,3))
                    #print(clusterlayers.shape)
                    if args.clustersave:
                        cdict[str(nc)][I] = clusterlayers


                if b_idx < 10 and args.viz: #"VISUALIZE"
                    viz = np.concatenate(vizs, axis=0)[:,:64*128]
                    img = Image.fromarray(np.uint8(255*viz))
                    draw = ImageDraw.Draw(img)
                    for i, word in enumerate(text):
                        begin = 2
                        x, y = 0, (begin+i*2)*img.height/(2*len(text)+begin)
                        draw.text((x, y), word, fill= (255,255,255), font=font)
                    # for i, value in enumerate(pred[negatives].tolist()):
                    #     x, y = int(i*img.width/len(pred[negatives])), int(1+img.height/3)
                    #     draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)
                    # for i, value in enumerate(mergevalue.tolist()):
                    #     x, y = int(i*img.width/len(mergevalue)), int(1+2*img.height/3)
                    #     draw.text((x, y), str(round(value, 3)), fill=(255,255,255), font=self.font)

                    #plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", viz)
                    img.save(result_path+f"e{epoch}_b{b_idx}.png")

                if args.savekmeans:
                    break
        if args.clustersave:
            for key in cdict:
                print(key, cdict[key].shape)
                with gzip.GzipFile(self.data_path+f"{mode}-{key}-cluster", 'wb') as fp:
                    pickle.dump(cdict[key], fp)

    def create_patch_embedding_clusters(self):
        print("Starting to create patch embedding clusters with tree prob")
        args = self.args
        # HYPERARAMS
        patchwid = self.args.embed_patch_width
        stride = 2
        embed_dim = self.args.embed_dim
        n_clusters = self.args.embed_cluster
        reward_idx = 4
        n_samples = self.args.embed_train_samples
        channels = [0,1] if not args.hue else [0]

        self.embedder = PatchEmbedder(embed_dim=embed_dim, n_cluster=n_clusters,
                                      channels=channels, pw=patchwid, stride=stride)

        # REAL DATASET
        if not args.dummy:
            # LOAD NAV DATA
            navdatadir = "./data/navigate/"
            if not os.path.exists(navdatadir+"data.pickle"):
                self.collect_navigation_dataset(datadir=navdatadir)

            with gzip.open(navdatadir+"data.pickle", 'rb') as fp:
                NX, NY = pickle.load(fp)
                NY = NY[:,0]
                NX = NX/255
            print("loaded navigation data:", NX.shape, NY.shape)

            # FUSE WITH TREE DATA
            high_reward = self.Y[:,reward_idx]>= self.args.embed_pos_threshold
            print("high reward frames in treechop dataset:", sum(high_reward))
            TX = self.X[high_reward]/255
            TY = np.ones(len(TX))
            navselection = np.random.randint(len(NX), size=n_samples)
            treeselection = np.random.randint(len(TY), size=n_samples)
            X = np.concatenate((TX[treeselection], NX[navselection]), axis=0)
            Y = np.concatenate((TY[treeselection], NY[navselection]), axis=0)
            print("fused dataset:", X.shape, Y.shape)

            # VIS FUSED DATASET
            fused_dir = f"saves/patchembed/{self.embed_data_args}-frames/"
            os.makedirs(fused_dir, exist_ok=True)
            RGB_X = hsv_to_rgb(X)
            shape = RGB_X.shape[:3]
            xmid = shape[2]/2
            ymid = shape[1]/2
            xslice = slice(int(xmid-shape[1]/10), math.ceil(xmid+shape[1]/10))
            yslice = slice(int(ymid-shape[2]/3), math.ceil(ymid+shape[2]/3))
            RGB_X[Y==1] *= 0.5
            RGB_X[Y==1, yslice, xslice] *= 2
            for idx in range(len(X)):
                plt.imsave(fused_dir+f"{'negative' if not Y[idx] else 'positive'}-{idx}.png", RGB_X[idx])

        # DUMMY DATASET
        if self.args.dummy:
            tree = cv2.cvtColor(cv2.imread("data/navigate/tree.png"), cv2.COLOR_BGR2RGB)
            nav = cv2.cvtColor(cv2.imread("data/navigate/nav.png"), cv2.COLOR_BGR2RGB)
            X = np.stack((tree,nav), axis=0)
            X = rgb_to_hsv(X/255)
            Y = np.array([1,0])
            print("using dummy dataset:", X.shape)

        # PIXEL CLUSTERS
        pixels = X.reshape(-1,3)[::10,channels]
        print("fitting pixel clusters (gmm) to pixels with shape:", pixels.shape)
        pixel_clusters = GMM(n_components=embed_dim).fit(pixels)
        self.embedder.pixel_clusters = pixel_clusters

        print("embedding the dataset...")
        flat_embeds, pshape = self.embedder.embed_batch(X)

        if False:
            # CREATE PATCHES
            print("creating patches...")
            patches = self.embedder.make_patches(X, patchwid, stride)
            print("patches shape and max:", patches.shape, np.max(patches))

            # CREATE EMBEDDINGS
            print("creating embeddings...")
            embeds = self.embedder.embed_patches(patches, verbose=True)

            # CLUSTER EMBEDDING SPACE
            print("clustering embedding space...")
            flat_embeds = embeds.reshape(-1, embed_dim)

        # CLUSTER PATCH EMBEDS
        skipped_embeds = flat_embeds[::5]
        #print("fitting the embedding clusters (gmm) on embeds with shape:", skipped_embeds.shape)
        #embed_clusters = GMM(n_components=n_clusters)
        print("fitting the embedding clusters (kmeans) on embeds with shape:", skipped_embeds.shape)
        embed_clusters = KMeans(n_clusters=n_clusters)
        embed_clusters.fit(skipped_embeds)
        flat_labels = embed_clusters.predict(flat_embeds)
        labels = flat_labels.reshape(pshape[0:3])

        # CALC CLUSTER TREE PROBABILITIES
        print("calculating cluster tree probs...")
        #gt = np.ones(embeds.shape[:3])*Y[:,None,None]
        shape = pshape[:3]
        gt = np.zeros(shape)
        xmid = shape[2]/2
        ymid = shape[1]/2
        xslice = slice(int(xmid-shape[1]/10), math.ceil(xmid+shape[1]/10))
        yslice = slice(int(ymid-shape[2]/3), math.ceil(ymid+shape[2]/3))
        gt[Y==1, yslice, xslice] = 1
        flat_gt = gt.reshape(-1)
        tree_probs = np.zeros(n_clusters)
        for idx in range(n_clusters):
            flat_patch_selection = flat_labels == idx
            tree_probs[idx] = np.sum(flat_gt[flat_patch_selection])/np.sum(flat_patch_selection)


        # SAVE CLUSTERS AND PROBS
        print("cluster probs:", tree_probs)
        self.embedder.patch_embed_clusters = embed_clusters
        self.embedder.patch_embed_cluster_tree_probs = tree_probs

        os.makedirs(self.embed_data_path, exist_ok=True)
        with open(self.embed_data_path+self.embed_data_args+".pickle", "wb") as fp:
            pickle.dump((embed_clusters, tree_probs, self.args.embed_dim, pixel_clusters), fp)

        print("Finished creating patch embedding clusters with tree probs")

    def vis_embed(self):
        resultdir = f"./results/patch-embed/result-videos-3/"
        result_args = f"{self.embed_data_args}"
        os.makedirs(resultdir, exist_ok=True)

        # LOAD CLUSTERS AND PROBS
        embed_tuple_path = self.embed_data_path+self.embed_data_args+".pickle"
        print(embed_tuple_path)
        if not os.path.exists(embed_tuple_path):
            print("no clusters and probs found...")
            self.create_patch_embedding_clusters()
        else:
            print("found clusters and probs...")
            self.embedder = PatchEmbedder(self.args.embed_dim, self.args.embed_cluster,
                                          pw=self.args.embed_patch_width,
                                          channels = [0] if self.args.hue else [0,1])
            self.embedder.load_embed_tuple(embed_tuple_path)

        # GET DATA
        if self.args.dummy:
            tree = cv2.cvtColor(cv2.imread("data/navigate/tree.png"), cv2.COLOR_BGR2RGB)
            nav = cv2.cvtColor(cv2.imread("data/navigate/nav.png"), cv2.COLOR_BGR2RGB)
            X = np.stack((tree,nav), axis=0)
            X = rgb_to_hsv(X/255)
        else:
            X = self.XX[:1000]/255

        if False:
            # MAKE PATCHES
            patches = self.embedder.make_patches(X, 8, 2)
            print("patches shape:",patches.shape)

            # CALC PROBS
            probs = self.embedder.calc_tree_probs_for_patches(patches, verbose=True)
            print("probs shape:", probs.shape)

        print("embedding batch...", X.shape)
        probs = self.embedder.predict_batch(X, verbose=True)
        print("probs shape:", probs.shape)


        rgb = hsv_to_rgb(X)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(resultdir + result_args + '.avi', fourcc, 20.0, (64 * 4, 64))
        for idx, frame in enumerate(probs):
            print("visualizing results, at frame:", idx, "/", len(probs), end='\r')
            resized_frame = np.ones((64,64,3)) * cv2.resize(frame, (64,64))[:,:,None]
            clean_mask = resized_frame>0.7
            masked_rgb = rgb[idx]*resized_frame
            pic = np.concatenate((rgb[idx], masked_rgb, resized_frame, clean_mask), axis=1)
            #plt.imsave(resultdir+f"{idx}.png", pic)
            uint8_bgr = cv2.cvtColor((255*pic).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(uint8_bgr)
        out.release()

    def vis_pixels(self):
        # GET PIXELS
        #data = self.X[self.Y[:,3]>0.9]
        #data = data[:,10:54,26:39]
        data = self.X
        #navdatadir = "./data/navigate/"
        #with gzip.open(navdatadir+"data.pickle", 'rb') as fp:
        #    data, NY = pickle.load(fp)
        #print(data.shape)
        pixels = data.reshape(-1,3)

        # PLOTS SETUP
        my_cmap = copy.copy(cm.get_cmap('plasma'))
        my_cmap.set_bad(my_cmap.colors[0])
        #print(cm.cmaps_listed)
        hs_pic = np.array([[[h,s,1] for s in range(255)] for h in range(255)])
        hs_pic = 255*hsv_to_rgb(hs_pic/255)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        ax3.set_aspect(1)

        ax2.imshow(hs_pic)
        ax2.invert_yaxis()
        ax1.hist2d(pixels[:,0], pixels[:,1], bins=100, norm=colors.LogNorm(), cmap=my_cmap)
        #plt.gca().set_aspect('equal', adjustable='box')

        #plt.show()
        X = pixels[::200,:2]
        print(X.shape)
        comps = 100
        gmm = GMM(n_components=comps)
        labels = gmm.fit_predict(X)
        pixperlabel = [np.sum(labels==i)/len(labels) for i in range(comps)]
        normed_ppl_perpix = (pixperlabel/max(pixperlabel))[labels]
        print(sorted(pixperlabel))
        ax3.scatter(X[:, 0], X[:, 1], c=labels, s=0.5, cmap='jet')
        ax3.set_xlim(0,255)
        ax3.set_ylim(0,255)
        plt.tight_layout()
        plt.show()

    def collect_split_dataset(self, path, size=2000, wait=10, test=0, datadir="./results/stuff/"):
        args = self.args
        os.makedirs(datadir+"samples/", exist_ok=True)
        # os.environ["MINERL_DATA_ROOT"] = "./data"
        if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRLTreechopVectorObf-v0"):
            minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment='MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                                num_workers=args.workers[0], worker_batch_size=args.workers[1])
        X = []
        Y = []
        cons = self.args.cons
        wait = self.args.wait
        delay = self.args.delay
        warmup = self.args.warmup
        chunksize = self.args.chunksize

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(test or 10,
                                                2*wait if not test else size, preload_buffer_size=args.workers[2])):
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
                            #draw.text((x, y), "\n".join([str(round(entry,3))
                            # for entry in rewtuple]), fill= (255,255,255), font=self.font)
                            img.save(datadir+"samples/"+f"{b_idx}-{chopidx}-{chunkidx}-"+
                                                        f"{'A' if rewards[chopidx*effchsize+chunkidx] else 'B'}.png")

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

    def collect_navigation_dataset(self, datadir="./data/navigate/"):
        print("Collecting nav dataset...")
        os.makedirs(datadir+"samples/", exist_ok=True)
        os.environ["MINERL_DATA_ROOT"] = "./data"
        minerl.data.download("./data", experiment='MineRLNavigateVectorObf-v0')
        #data = minerl.data.make('MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLNavigateVectorObf-v0',
            data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
            num_workers=args.workers[0],
            worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()

        # INIT STRUCTURES
        X = []
        Y = []
        n_per_chunk = 10
        skip = 100

        # ITER OVER EPISODES
        for fridx, name in enumerate(names):
            # EXTRACT EPISODE
            state, action, reward, state_next, done = zip(*data.load_data(name))

            # CONVERT COLOR
            pov = np.array([s['pov'] for s in state])
            if self.args.color == "HSV":
                pov = (255*rgb_to_hsv(pov/255)).astype(np.uint8)


            selections = [pov[skip*i:skip*i+n_per_chunk] for i in range(len(pov)//skip)]
            #rewards = np.zeros((len(selections), 1))

            if len(X)<300:
                for chidx, chunk in enumerate(selections):
                    for fi, frame in enumerate(chunk):
                        img = Image.fromarray(np.uint8(255*hsv_to_rgb(frame/255)))
                        draw = ImageDraw.Draw(img)
                        #print(rewards)
                        rewtuple = (0,)
                        x, y = 0, 0
                        draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]),
                                  fill= (255,255,255), font=self.font)
                        img.save(datadir+"samples/"+f"{name}-{chidx}-{fi}.png")

            episode_chunks = []
            for chunk in selections:
                episode_chunks.extend(chunk)
            X.extend(episode_chunks)
            if len(X)>1000:
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.zeros((X.shape[0], 1))

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(datadir+"data.pickle", 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_discounted_dataset(self, path, size=2000, datadir="./results/stuff/", test=0):
        args = self.args
        os.makedirs(datadir+"samples/", exist_ok=True)
        os.environ["MINERL_DATA_ROOT"] = "./data"
        #minerl.data.download("./data", experiment='MineRLTreechopVectorObf-v0')
        #data = minerl.data.make('MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0',
            data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
            num_workers=args.workers[0],
            worker_batch_size=args.workers[1])
        names = data.get_trajectory_names()
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        delta = self.args.delta
        gamma = self.args.gamma
        revgamma = self.args.revgamma
        trajsize = self.args.trajsize
        if test:
            testsize = size
            size = size*test

        print("collecting data set with", size, "frames")
        #for b_idx, (state, act, reward, next_state, done) in
        # enumerate(data.batch_iter(test or 10, cons if not test else testsize, preload_buffer_size=args.workers[2])):
        for fridx, name in enumerate(names):
            # EXTRACT EPISODE
            state, action, reward, state_next, done = zip(*data.load_data(name))

            # CONVERT COLOR
            pov = np.array([s['pov'] for s in state])
            if self.args.color == "HSV":
                pov = (255*rgb_to_hsv(pov/255)).astype(np.uint8)

            # DETECT AND FILTER CHOPS
            chops = np.nonzero(reward)[0]
            deltas = chops[1:]-chops[:-1]
            big_enough_delta = deltas>50
            chops = np.concatenate((chops[None,0], chops[1:][big_enough_delta]))
            #print(chops)

            # INIT EPISODE SET
            approaches = []
            rewards = []

            # VERIFY CHOPS AND SEQUENCES
            if chops.size ==0:
                continue
            end = np.max(chops)
            sequ = pov[:end+1]
            reward = reward[:end+1]
            assert reward[-1]>0, "ERROR wrong chop detection"

            # INIT DISCOUNT
            delaycount = delay
            rowrew = []
            selection = []
            addfak = 0
            revaddfak = 0
            relchopidx = 0
            chopidx = -1

            # DISCOUNT LOOP
            for i in range(1, len(reward)+1):
                delaycount -= 1
                relchopidx -= 1

                # RESET
                if reward[-i]>0:
                    if len(reward)+i==chops[chopidx]:
                        relchopidx = 0
                        chopidx -= 1
                    fak = 1 #exponential
                    sub = 0 #subtraction
                    addfak += 1 #exponanential with add-reset
                    revfak = 1
                    revaddfak += 1
                    revhelper = 0.01
                    #fak = 0
                    delaycount = delay

                # DELAY AND TRAJECTORY SKIP
                if delaycount>0 or relchopidx <= -trajsize-delay:
                    continue

                # STORE REWARDS AND INDEXES
                selection.append(-i)
                rewtuple = (relchopidx, fak, addfak, revfak, revaddfak, sub)
                rowrew.append(rewtuple)

                # DISCOUNT FAKTORS
                fak *= gamma
                sub -= 1
                addfak *= gamma
                revfak = max(revfak-revhelper, 0)
                revaddfak = max(revaddfak-revhelper, 0)
                revhelper *= revgamma

            # EXTEND EPISODE SET
            #print(row)
            rewards.extend(rowrew[::-1])
            approaches.extend(sequ[selection[::-1]])

            # SAVE SAMPLE IMGS
            if len(X)<300:
                for fi, frame in enumerate(approaches):
                    img = Image.fromarray(np.uint8(255*hsv_to_rgb(frame/255)))
                    draw = ImageDraw.Draw(img)
                    #print(rewards)
                    rewtuple = rewards[fi]
                    x, y = 0, 0
                    draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]),
                              fill= (255,255,255), font=self.font)
                    img.save(datadir+"samples/"+f"{name}-{fi}.png")

            # EXTEND FULL DATA SET
            if approaches:
                X.extend(approaches)
                Y.extend(rewards)

            # QUIT IF SIZE REACHED
            if len(X) >= size:
                X = X[:size]
                Y = Y[:size]
                break

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)

        # SAVE AS ZIPPED FILE
        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

    def collect_discounted_dataset_old(self, path, size=2000, datadir="./results/stuff/", test=0):
        os.makedirs(datadir+"samples/", exist_ok=True)
        os.environ["MINERL_DATA_ROOT"] = "./data"
        #minerl.data.download("./data", experiment='MineRLTreechopVectorObf-v0')
        #data = minerl.data.make('MineRLTreechopVectorObf-v0')
        data = minerl.data.make('MineRLTreechopVectorObf-v0',
            data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
            num_workers=args.workers[0],
            worker_batch_size=args.workers[1])
        X = []
        Y = []
        cons = self.args.cons
        delay = self.args.delay
        delta = self.args.delta
        if test:
            testsize = size
            size = size*test

        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(
                data.batch_iter(test or 10, cons if not test else testsize, preload_buffer_size=args.workers[2])):
            print("at batch", b_idx, end='\n')
            #vector = state['vector']

            # CONVERT COLOR
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
                deltas = chops[1:]-chops[:-1]
                big_enough_delta = deltas>50
                chops = np.concatenate((chops[None,0], chops[1:][big_enough_delta]))
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
                rewards.extend(rowrew[::-1])
                approaches.extend(sequ[selection[::-1]])

            if len(X)<300: # SAVE IMG
                for fi, frame in enumerate(approaches):
                    img = Image.fromarray(np.uint8(255*hsv_to_rgb(frame/255)))
                    draw = ImageDraw.Draw(img)
                    #print(rewards)
                    rewtuple = rewards[fi]
                    x, y = 0, 0
                    draw.text((x, y), "\n".join([str(round(entry,3)) for entry in rewtuple]),
                              fill= (255,255,255), font=self.font)
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

        # CONVERT TO FINAL ARRAYS
        X = np.array(X, dtype=np.uint8)
        Y = np.array(Y)

        with gzip.GzipFile(path, 'wb') as fp:
            pickle.dump((X, Y), fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-final", action="store_true")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-unet", action="store_true")
    parser.add_argument("-live", action="store_true")
    parser.add_argument("-resnet", action="store_true")
    parser.add_argument("-critic", action="store_true")
    parser.add_argument("-distnorm", action="store_true")
    parser.add_argument("-uload", action="store_true")
    parser.add_argument("-cload", action="store_true")
    parser.add_argument("-dream", action="store_true")
    parser.add_argument("-discounted", type=bool, default=True)
    parser.add_argument("-sigmoid", action="store_true")
    parser.add_argument("-clustersave", action="store_true")
    parser.add_argument("-staticL1", action="store_true")
    parser.add_argument("-savekmeans", action="store_true")
    parser.add_argument("-integrated", action="store_true")
    parser.add_argument("-grounded", action="store_true")
    parser.add_argument("-clippify", action="store_true")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-dummy", action="store_true")
    parser.add_argument("-grid", action="store_true")
    parser.add_argument("-hue", action="store_true")
    #parser.add_argument("-vizdataset", action="store_true")

    parser.add_argument("--blur", type=int, default=0)
    parser.add_argument("--cluster", type=str, default="")
    parser.add_argument("--clustercritic", type=int, default=0)
    parser.add_argument("--trajsize", type=int, default=50)
    parser.add_argument("--gray", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--testsize", type=int, default=300)
    parser.add_argument("--dreamsteps", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--L2", type=float, default=0.0)
    parser.add_argument("--L1", type=float, default=30.0)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--rewidx", type=int, default=3)
    parser.add_argument("--wait", type=int, default=120)
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--revgamma", type=float, default=1.1)
    parser.add_argument("--delta", type=int, default=50)
    parser.add_argument("--datasize", type=int, default=10000)
    parser.add_argument("--chunksize", type=int, default=20)
    parser.add_argument("--cons", type=int, default=250)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--embed-cluster", type=int, default=10)
    parser.add_argument("--embed-train-samples", type=int, default=1000)
    parser.add_argument("--embed-patch-width", type=int, default=8)
    parser.add_argument("--embed-pos-threshold", type=float, default=0.9)
    parser.add_argument("--color", type=str, default="HSV")
    parser.add_argument("--name", type=str, default="default")
    args = parser.parse_args()
    args.workers = (1,1,1)
    print(args)

    H = Handler(args)
    H.load_data()
    #H.vis_pixels()
    try:
        H.create_patch_embedding_clusters()
        H.vis_embed()
    except Exception as e:
        print("ERROR", e)
        resultdir = f"./results/patch-embed/"
        result_args = f"{H.embed_data_args}"
        os.makedirs(resultdir, exist_ok=True)
        with open(resultdir + result_args + "-fail.txt", 'w') as fp:
            fp.write(str(e))

    if args.debug:
        #H.patch_embedding([])
        pass
    try:
        if args.cluster:
            if args.train or args.savekmeans:
                H.cluster(mode="train")
            if args.test:
                H.cluster(mode="test")
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
                H.load_models(modelnames=[H.criticname])
            H.dream()

    except Exception as e:
        L.exception("Exception occured:"+ str(e))
        print("EXCEPTION")
        print(e)