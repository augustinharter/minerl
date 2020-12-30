import math
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture as GMM


class PatchEmbedder():
    def __init__(self, embed_dim=100, n_cluster=100, pw=8, stride=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_cluster = n_cluster
        self.patch_embed_clusters = None
        self.patch_embed_cluster_tree_probs = None
        self.step_faktor = round(math.sqrt(self.embed_dim))
        self.pixel_clusters : GMM = None
        self.w = pw
        self.s = stride

    def load_embed_tuple(self, embed_tuple_path):
        with open(embed_tuple_path, "rb") as fp:
            self.patch_embed_clusters, self.patch_embed_cluster_tree_probs, self.embed_dim, self.pixel_clusters = pickle.load(fp)
            self.step_faktor = round(math.sqrt(self.embed_dim))
            self.n_cluster = len(self.patch_embed_cluster_tree_probs)

    def embed_patch(self, patchpixels):
        # CALC INDEXES
        flat_pixels = patchpixels.reshape(-1,3)
        #print(flat_pixels)

        #embed = np.histogram(refactored_pixels, bins=np.linspace(0, self.step_faktor+1, self.embed_dim+1))[0]
        if self.pixel_clusters is None:
            hist2d, _, _ = np.histogram2d(flat_pixels[:,1], flat_pixels[:,0], range=((0,1),(0,1)), bins=self.step_faktor)
            #print(hist2d.shape, hist2d)
            embed = hist2d.reshape(-1)
            #print(embed.shape, embed)
            #print(np.sum(embed))
        else:
            labels = self.pixel_clusters.predict(flat_pixels[:,:2])
            embed = np.array([np.sum(labels==i) for i in range(self.embed_dim)])

        #embed = embed/np.linalg.norm(embed)
        #print(self.step_faktor)
        #print(refactored_pixels.shape, np.sum(embed), embed)
        return embed

    def make_patches(self, x, w, stride):
        if not stride:
            stride = w

        #xpad = math.ceil(x.shape[2]/w)*w - x.shape[2]
        #ypad = math.ceil(x.shape[1]/w)*w - x.shape[1]
        #padding =  [(0,0), (math.ceil(ypad/2), ypad//2),(math.ceil(xpad/2), xpad//2), (0,0)]
        #if type(x)==type(np.array([])):
        #    x = np.pad(x,padding, mode="edge")
        #else:
        #    print("ERROR | didnt receive np-array!")

        all_tiles = []
        for sample in x:
            tiles = [sample[stride*i:stride*i+w,stride*j:stride*j+w]
                for i in range(1+(sample.shape[0]-w)//stride)
                for j in range(1+(sample.shape[1]-w)//stride)]
            tiles = np.stack(tiles, axis=0)
            all_tiles.append(tiles)

        patches = np.stack(all_tiles, axis=0)
        patched_width = int(math.sqrt(patches.shape[1]))
        patches = patches.reshape(patches.shape[0], patched_width, patched_width, w, w, 1 if len(x.shape)==3 else x.shape[-1])
        return patches

    def embed_patches(self, patches, verbose=False):
        patched_xwid = patches.shape[2]
        patched_ywid = patches.shape[1]
        embeds = np.zeros(patches.shape[:3]+(self.embed_dim,))
        print(f"embedding patches  of frame {p} out of {len(embeds)}", end='\r')
        for p in range(len(embeds)):
            if verbose:
                pass
            for y in range(patched_ywid):
                for x in range(patched_xwid):
                    embeds[p, y, x] = self.embed_patch(patches[p,y,x])
        if verbose:
            print()
        return embeds

    def calc_tree_probs_for_patches(self, patches, verbose=False):
        embed_dim = self.embed_dim

        # EMBED PATCHES
        if verbose:
            print("embedding patches...")
        embeds = self.embed_patches(patches, verbose=verbose)
        flat_embeds = embeds.reshape(-1, embed_dim)

        # PREDICT EMBEDS
        if verbose:
            print("predicting cluster labels...")
        flat_labels = self.patch_embed_kmeans.predict(flat_embeds)

        # GET TREE PROB FOR LABELS
        if verbose:
            print("getting patch probs...")
        cluster_probs = self.patch_embed_cluster_tree_probs/np.max(self.patch_embed_cluster_tree_probs)
        flat_tree_probs = cluster_probs[flat_labels]
        tree_probs = flat_tree_probs.reshape(embeds.shape[:3])

        return tree_probs

    def embed_batch(self, batch:np.ndarray, verbose=False):
        # SHAPING
        bshape = batch.shape
        flat_batch = batch.reshape(-1, 3)
        flat_hs = flat_batch[:, :2]

        # PIXEL EMBEDS (BATCHED)
        if verbose:
            print("predicting pixel cluster labels...")
        flat_labels = []
        bs = 100000
        for bi in range(1+len(flat_hs)//bs):
            flat_labels.append(self.pixel_clusters.predict(flat_hs[bi*bs:(bi+1)*bs]))
        flat_labels = np.concatenate(flat_labels, axis=0)
        labels = flat_labels.reshape(bshape[:-1])

        # PATCHING
        label_patches = self.make_patches(labels, self.w, self.s)
        pshape = label_patches.shape
        #print("labelpatches", pshape)

        # PATCH EMBEDS
        #flat_label_patches = label_patches.reshape(-1, self.w, self.w)
        flat_label_patches = label_patches.reshape(-1, self.w*self.w)
        flat_embeds = np.zeros((len(flat_label_patches), self.embed_dim))

        if verbose:
            print("embedding patches...")
        for label in range(self.embed_dim):
           flat_embeds[:,label] = np.sum(flat_label_patches==label, axis=1)
        #for pidx in range(len(flat_label_patches)):
        #    flat_embeds[pidx] = self.label_patch_to_embed(flat_label_patches[pidx])

        return flat_embeds, pshape

    def predict_batch(self, batch : np.ndarray, verbose=False):
        # EMBED BATCH
        flat_embeds, pshape = self.embed_batch(batch, verbose=verbose)

        # LABEL EMBEDS
        if verbose:
            print("predicting patch cluster labels...")
        flat_labels = self.patch_embed_clusters.predict(flat_embeds)

        # GET TREE PROB FOR LABELS
        if verbose:
            print("getting patch probs...")
        cluster_probs = self.patch_embed_cluster_tree_probs / np.max(self.patch_embed_cluster_tree_probs)
        flat_tree_probs = cluster_probs[flat_labels]
        tree_probs = flat_tree_probs.reshape(pshape[:3])

        return tree_probs

    def label_patch_to_embed(self, patch):
        return np.array([np.sum(patch == i) for i in range(self.embed_dim)])

    def calc_tree_probs_for_batch(self, batch, verbose=False):
        embed_dim = self.embed_dim

        # EMBED BATCH
        if verbose:
            print("embedding patches...")
        embeds = self.embed_batch(batch, verbose=verbose)
        flat_embeds = embeds.reshape(-1, embed_dim)

        # PREDICT EMBEDS
        if verbose:
            print("predicting cluster labels...")
        flat_labels = self.patch_embed_kmeans.predict(flat_embeds)

        # GET TREE PROB FOR LABELS
        if verbose:
            print("getting patch probs...")
        cluster_probs = self.patch_embed_cluster_tree_probs / np.max(self.patch_embed_cluster_tree_probs)
        flat_tree_probs = cluster_probs[flat_labels]
        tree_probs = flat_tree_probs.reshape(embeds.shape[:3])

        return tree_probs