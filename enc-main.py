import argparse
from matplotlib import pyplot as plt
import utils
import os
from nets import AutoEncoder, VAE
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
        self.color_dim = 1 if args.gray else args.colorchs
        self.AE = VAE(args.tilewid, args.encdim, self.color_dim) if not args.classical else AutoEncoder(args.tilewid, args.encdim, self.color_dim)
        self.models = dict()
        self.models["auto-encoder"] = self.AE
        self.arg_path = f"en{args.encdim}-tw{args.tilewid}-st{args.stride}-{'gray' if args.gray else f'cd{args.colorchs}'}/"
        print("model path:", self.arg_path)
        self.result_path = f"./results/VAE/"+ args.name+ "-"+ self.arg_path
        print("viz path:", self.result_path)
        self.save_path = f"./saves/VAE/"+self.arg_path

    def load_data(self, data_size = 10000, batch_size = 8):
        tile_wid = self.args.tilewid
        stride = self.args.stride
        data_path = f"data/tiles/tree-chop/w{tile_wid}-s{stride}-n{data_size}/"
        file_name = "data.pickle"
        cons = 20
        test_size = 10

        print("loading data:", data_path)
        if not os.path.exists(data_path+file_name):
            print("train set...")
            os.makedirs(data_path, exist_ok=True)
            self.collect_tiled_dataset(data_path+file_name, size=data_size)
        if not os.path.exists(data_path+"test-"+file_name):
            print("collecting test set...")
            os.makedirs(data_path, exist_ok=True)
            self.collect_tiled_dataset(data_path+"test-"+file_name, size=test_size*cons, cons=cons)

        # TRAIN data
        with open(data_path+file_name, "rb") as fp:
            self.pad_shape, self.data = pickle.load(fp)
        self.dataloader = [self.data[b*batch_size:(b+1)*batch_size] for b in range(math.ceil(len(self.data)/batch_size))]
        print("padded shape of one frame", self.pad_shape[1:])
        print(f"loaded train set with {len(self.data)} frames with each {len(self.data[0])} tiles")

        self.tiles_per_frame = len(self.data[0])

        # TEST data
        batch_size = cons
        with open(data_path+"test-"+file_name, "rb") as fp:
            self.test_pad_shape, self.test_data = pickle.load(fp)
        self.testdataloader = [self.test_data[b*batch_size:(b+1)*batch_size] for b in range(math.ceil(len(self.data)/batch_size))]
        print(f"loaded test set with {len(self.test_data)} frames with each {len(self.test_data[0])} tiles")

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
        AE = self.AE
        args = self.args
        test = mode=="test"
        train = mode=="train"
        loader = self.dataloader if train else self.testdataloader[1:]

        # Setup save path and Logger
        result_path = self.result_path+mode+"/" 
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{self.args}\n\n")

        # Epoch and Batch Loops
        for epoch in range(int(test) or self.args.epochs):
            for b_idx, batch in enumerate(loader):

                # FORWARD PASS---------------------------
                if test: # Stop early if testing
                    if b_idx>=10:
                        break

                flat_batch = batch.reshape(-1, args.tilewid, args.tilewid, batch.shape[-1])
                if args.colorchs<3 and not args.gray: # format color channels
                    cflat_batch = flat_batch[:,:,:,:args.colorchs]
                    ttiles = T.from_numpy(cflat_batch).float()
                elif args.gray:
                    cflat_batch = flat_batch[:,:,:,None,2]
                    ttiles = T.from_numpy(cflat_batch).float()
                else:
                    ttiles = T.from_numpy(flat_batch).float()

                if train:
                    loss, res, enc = AE.train_batch(ttiles) # Training AE with batch               
                    log_msg = f"TRAIN | epoch {epoch} | batch {b_idx} | loss: {loss}"

                if test:
                    res, enc = AE.test_batch(ttiles) # Eval AE with batch  
                    log_msg = f"TEST | epoch {epoch} | batch {b_idx}"

                print(log_msg, end="\r")
                log_file.write(log_msg+"\n")
                # -------------------------------------

                # VISUALIZE----------------------------
                if epoch%2 and not b_idx or test: 

                    # ORIGINAL AND RECONSTRUCTED
                    retiled_batch = self.retile(flat_batch)
                    if args.colorchs<3 and not args.gray:
                        cretiled_res = self.retile(res.numpy())
                        #retiled_res = np.concatenate((cretiled_res, retiled_batch[:,:,:,args.colorchs:]))
                        #value_channel = np.ones_like(retiled_batch[:,:,:,args.colorchs:])
                        value_channel = retiled_batch[:,:,:,args.colorchs:]
                        retiled_res = np.concatenate((cretiled_res, value_channel), axis=-1)
                    else:
                        retiled_res = self.retile(res.numpy())

                    diff = retiled_res
                    if args.gray: # Add gray original
                        diff = np.concatenate((retiled_batch[:,:,:,None,2], diff), axis=2)
                    #print("max values", np.max(batch), np.max(retiled))
                    #print("min values", np.min(batch), np.min(retiled))

                    if args.gray:
                        diff = np.concatenate((np.zeros_like(diff), np.zeros_like(diff), diff), axis=-1)
                    diff = np.concatenate((retiled_batch, diff), axis=2) # Add color original


                    # COLOR TILES if encoding has 2 or 3 dimensions
                    if False:
                        if enc.shape[1]==2:
                            pos_enc = T.sigmoid(enc)
                            enc_multi = T.stack((pos_enc[:,0], pos_enc[:,1], 0.75*T.ones_like(pos_enc[:,1])), dim=1)[:,None,None]
                            vis_enc = T.ones(*res.shape[:-1], 3) * enc_multi
                        elif enc.shape[1]==3:
                            pos_enc = T.sigmoid(enc)
                            vis_enc = T.ones(*res.shape[:-1], 3) * T.stack((pos_enc[:,0], pos_enc[:,1], pos_enc[:,2]), dim=1)[:,None,None]
                        if enc.shape[1]==3 or enc.shape[1]==2:
                            retiled_enc = self.retile(vis_enc.numpy())
                            diff = np.concatenate((diff, retiled_enc), axis=2)


                    # CONVERT HSV TO RGB
                    diff = hsv_to_rgb(diff)
                    
                    #SCATTERPLOTS
                    if enc.shape[1]==3 or enc.shape[1]==2:

                        # PIXEL SCATTERS
                        if args.cmode == "HS":
                            chs = [0,1]
                        if args.cmode == "HV":
                            chs = [0,2]
                        n_clusters = args.clusters
                        per_frame_viz = []
                        for frame in range(-1, len(batch)):
                            if args.pool:
                                #print(batch.shape)
                                batch[:,:,:,:,0] = F.avg_pool2d(T.from_numpy(batch[:,:,:,:,0]), 3, stride=1, padding=1, count_include_pad=False).numpy()
                            if frame == -1:
                                frencs = batch.reshape(-1,3)
                                frencs[:,chs[1]] *= args.squish # format second dimension
                                kmeans =  [KMeans(n_clusters=n).fit(frencs[:,chs]) for n in [2,3,5,10]]
                                
                                if args.split:
                                    splits = []
                                    for base in kmeans:
                                        comb_labels = base.labels_
                                        target = np.array([0.12, 0.02])
                                        centers = base.cluster_centers_
                                        dist = (centers - target)**2
                                        dist = np.sum(dist, axis=1)
                                        nearest_center_idx = np.argmin(dist)
                                        #nearest_center = centers[nearest_center_idx]
                                        selection = comb_labels == nearest_center_idx
                                        splits.append((nearest_center_idx, KMeans(n_clusters=2).fit(frencs[selection][:,chs])))

                                continue
                            else:
                                shaped_frencs = batch[frame]
                                frencs = shaped_frencs.reshape(-1,3)
                                #reshape_test = frencs.reshape(batch[frame].shape)
                                #print((reshape_test==shaped_frencs).all())
                                #print(shaped_frencs.shape)
                                frencs[:,chs[1]] *= args.squish
                            
                            per_cluster_viz = []
                            for c_idx, clusters in enumerate(kmeans):
                                flat_labels = clusters.predict(frencs[:,chs]) # exctract and reshape labels

                                if args.split:
                                    selection = flat_labels == splits[c_idx][0]
                                    split_labels = splits[c_idx][1].predict(frencs[:,chs])
                                    flat_labels[selection==False] = 0
                                    flat_labels[selection] = split_labels[selection]+1
                                    flat_labels = flat_labels/3.0
                                else:
                                    flat_labels = flat_labels/(clusters.n_clusters)


                                # shape to overlay:
                                labels = flat_labels.reshape(batch[frame].shape[:-1])
                                #print(labels.shape)
                                labels = self.retile(labels[:,:,:,None])
                                #print(labels.shape)
                                seg = retiled_batch[frame].copy()
                                seg[:,:,0] = labels[0,:,:,0]
                                seg[:,:,1] = 1
                                overlay = hsv_to_rgb(seg)

                                fig = plt.figure()
                                if enc.shape[1]==2:
                                    ax = fig.add_subplot(111)
                                    ax.scatter(frencs[:,chs[0]], frencs[:,chs[1]], s=0.1, c=flat_labels)
                                    #plt.hist2d(frencs[:,0], frencs[:,2], 100)
                                if enc.shape[1]==3:
                                    ax = fig.gca(projection='3d')
                                    ax.scatter(frencs[:,0], frencs[:,1], frencs[:,2], s=0.1)

                                #plt.show()
                                
                                if not args.no_axnorm:
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(0,1)

                                if frame == 0:
                                    plt.savefig(result_path+f"e{epoch}_b{b_idx}_f{frame}_hsv-{clusters.n_clusters}.png")

                                ax.axis('off')
                                scatter = self.get_arr_from_fig(fig)
                                per_cluster_viz.append(np.concatenate((overlay, scatter), axis=1))
                                plt.close()

                            per_frame_viz.append(np.concatenate(per_cluster_viz, axis=1))
                        
                        overlay_and_scatters = np.stack(per_frame_viz, axis=0)
                        diff = np.concatenate((diff, overlay_and_scatters), axis=2)

                        # ENCODING SCATTERS
                        scatters = []
                        for frame in range(-1, len(batch)):
                            frencs_len = len(batch[frame])
                            if frame == -1:
                                frencs = enc
                            else:
                                frencs = enc[frame*frencs_len:(1+frame)*frencs_len]
                            #frencs = frencs / ((frencs.max()-frencs.min())/2)
                            #print("num encodings out of scope:", (frencs>1).sum()+(frencs<-1).sum())
                            #print(frencs_len)
                            fig = plt.figure()
                            if enc.shape[1]==2:
                                ax = fig.add_subplot(111)
                                #ax.scatter(frencs[:,0], frencs[:,1], s=0.3)
                                #ax.hexbin(frencs[:,0], frencs[:,1], cmap=cm.jet)
                                ax.hist2d(frencs[:,0].numpy(), frencs[:,1].numpy(), 100)#, cmap=cm.jet)
                            if enc.shape[1]==3:
                                ax = fig.gca(projection='3d')
                                ax.scatter(frencs[:,0], frencs[:,1], frencs[:,2], s=0.3)
                            
                            #ax.set_xlim(-1,1)
                            #ax.set_ylim(-1,1)
                            #plt.show()
                            if frame == -1:
                                plt.savefig(result_path+f"e{epoch}_b{b_idx}_comb-enc-scatter.png")
                            elif frame== 0:
                                plt.savefig(result_path+f"e{epoch}_b{b_idx}_f0-enc-scatter.png")

                            ax.axis('off')
                            if frame>=0:
                                scatters.append(self.get_arr_from_fig(fig))
                            plt.close()
                        scatters = np.stack(scatters, axis=0)
                        diff = np.concatenate((diff, scatters), axis=2)


                    # SAVE IMAGE    
                    diff = np.concatenate(diff, axis=0)
                    plt.imsave(result_path+f"e{epoch}_b{b_idx}.png", diff)
                # -----------------------------
            
            if epoch and not epoch%args.saveevery:
                self.save_models()
            
            if train:
                self.AE.sched.step()
 
        print()
        log_file.close()

    def test(self):
        data = self.data
        AE = self.AE
        args = self.args
        result_path = self.result_path+"test/"
        os.makedirs(result_path, exist_ok=True)
        log_file = open(result_path+"log.txt", "w")
        log_file.write(f"{args}\n\n")

        for b_idx, batch in enumerate(self.testdataloader):
            if b_idx>20:
                break
            flat_batch = batch.reshape(-1, args.tilewid, args.tilewid, 3)
            if args.colorchs<3:
                cflat_batch = flat_batch[:,:,:,:args.colorchs]
            ttiles = T.from_numpy(cflat_batch).float()

            res, enc = AE.test_batch(ttiles) # Training AE with batch

            # VISUALIZE
            retiled_res = self.retile(res.numpy())
            retiled_batch = self.retile(flat_batch)
            #print("max values", np.max(batch), np.max(retiled))
            #print("min values", np.min(batch), np.min(retiled))
            diff = np.concatenate((retiled_batch, retiled_res), axis=2)

            # Comparing POV and TILES
                #plt.imsave(f"./results/test/pov-to-tile/pov.png", pov[0,0])
                #for i in range(16*16):
                #    plt.imsave(f"./results/test/pov-to-tile/tile_{i}.png", tiles[i])
                #retiled = utils.retile(tiles, 64, 64, stride=0)
                #plt.imsave(f"./results/test/pov-to-tile/retiled-pov.png", retiled[0])

            # Color tiles if encoding has 2 or 3 dimensions
            if enc.shape[1]==2:
                vis_enc = T.ones_like(res) * T.stack((enc[:,0], 0*enc[:,0], enc[:,1]), dim=1)[:,None,None]
            if enc.shape[1]==3:
                vis_enc = T.ones_like(res) * T.stack((enc[:,0], enc[:,1], enc[:,2]), dim=1)[:,None,None]
            if enc.shape[1]==3 or enc.shape[1]==2:
                retiled_enc = self.retile(vis_enc.numpy())
                diff = np.concatenate((diff, retiled_enc), axis=2)
                
            diff = np.concatenate(diff, axis=0)
            plt.imsave(result_path+f"b{b_idx}.png", hsv_to_rgb(diff))

            if enc.shape[1]==2:
                plt.scatter(enc[:,0], enc[:,1])
            if enc.shape[1]==3:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(enc[:,0], enc[:,1], enc[:,2])
            if enc.shape[1]==3 or enc.shape[1]==2:
                plt.savefig(result_path+f"b{b_idx}-scatter.png")
                plt.close()
                
            print(f"EVAL | batch {b_idx}", end = "\r")
            #log_file.write(f"{loss}\n")

        log_file.close()
        print()
    
    def make_tiles(self, x):
        stride = self.args.stride
        w = self.args.tilewid

        if not stride:
            stride = w
        xpad = math.ceil(x.shape[2]/w)*w - x.shape[2]
        ypad = math.ceil(x.shape[1]/w)*w - x.shape[1]
        padding =  [(0,0), (math.ceil(ypad/2), ypad//2),(math.ceil(xpad/2), xpad//2), (0,0)]
        if type(x)==type(np.array([])):
            x = np.pad(x,padding, mode="edge")
        else:
            print("ERROR | didnt receive np-array!")
        all_tiles = []
        for sample in x:
            tiles = [sample[stride*i:stride*i+w,stride*j:stride*j+w] 
                for i in range(sample.shape[0]//stride)
                for j in range(sample.shape[1]//stride)]
            tiles = np.stack(tiles, axis=0)
            all_tiles.append(tiles)
        return np.stack(all_tiles, axis=0), x.shape

    def retile(self, tiles):
        stride = self.args.stride
        y, x = self.pad_shape[1], self.pad_shape[2]

        tile_w = tiles[0].shape[0]
        tile_x = x//tile_w
        tile_y = y//tile_w
        if not stride:
            stride = tile_w
        samples = []
        n_samples = len(tiles)//(tile_x*tile_y)
        for s_idx in range(n_samples):
            sample = np.zeros((y,x,tiles.shape[3]), dtype=tiles[0].dtype)
            for yy in range(tile_y):
                for xx in range(tile_x):
                    sample[yy*stride:yy*stride+tile_w, xx*stride:xx*stride+tile_w] = tiles[s_idx*(tile_x*tile_y)+yy*tile_x+xx]
            samples.append(sample)
        return np.stack(samples, axis=0)

    def collect_tiled_dataset(self, path, size=2000, cons=1):
        tile_wid = self.args.tilewid
        stride = self.args.stride
        gray = self.args.gray

        os.environ["MINERL_DATA_ROOT"] = "/home/augo/uni/minerl/data"
        data = minerl.data.make('MineRLTreechopVectorObf-v0')
        data_set = []
        print("collecting data set with", size, "frames")
        for b_idx, (state, act, rew, next_state, done) in enumerate(data.batch_iter(10,400, num_epochs=1)):
            
            print("at batch", b_idx, end='\r')
            #vector = state['vector']
            pov = state['pov']/255
            pov = rgb_to_hsv(pov)

            dist = 70
            wait = 30
            chops = [(i,pos) for (i,pos) in enumerate(np.argmax(rew, axis=1)) if pos>dist]
            #print(chops)
            approaches = []
            for chop in chops:
                approaches.append(pov[chop[0],chop[1]-dist:chop[1]-(dist-2*cons):2]) # take 30 frames from 50 frames before chop
            
            #print(approaches)
            if approaches:
                selection = np.stack(approaches, axis = 0)
                #print(selection.shape)
                tiles, shape = self.make_tiles(selection.reshape(-1,64,64,pov.shape[-1]))
                data_set.extend(tiles)

            if len(data_set) >= size:
                data_set = data_set[:size]
                break

        np_data = np.stack(data_set, axis=0)
        with open(path, "wb") as fp:
            pickle.dump((shape, np_data), fp)

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
    parser.add_argument("-split", action="store_true")
    parser.add_argument("-pool", action="store_true")
    parser.add_argument("-no-axnorm", action="store_true")
    parser.add_argument("-classical", action="store_true")
    parser.add_argument("--gray", type=bool, default=True)
    parser.add_argument("--encdim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--tilewid", type=int, default=8)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--colorchs", type=int, default=2)
    parser.add_argument("--saveevery", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cmode", type=str, default="HS")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--squish", type=float, default=0.2)
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
