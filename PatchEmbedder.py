import math
import numpy as np
import pickle


class PatchEmbedder():
    def __init__(self, embed_dim=100, n_cluster=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_cluster = n_cluster
        self.patch_embed_kmeans = None
        self.patch_embed_cluster_tree_probs = None
        self.step_faktor = round(math.sqrt(self.embed_dim))

    def load_embed_tuple(self, embed_tuple_path):
        with open(embed_tuple_path, "rb") as fp:
            self.patch_embed_kmeans, self.patch_embed_cluster_tree_probs, self.embed_dim = pickle.load(fp)
            self.step_faktor = round(math.sqrt(self.embed_dim))
            self.n_cluster = len(self.patch_embed_cluster_tree_probs)

    def embed_patch(self, patchpixels):
        # CALC INDEXES
        flat_pixels = patchpixels.reshape(-1,3)
        refactored_pixels = flat_pixels[:,0]*self.step_faktor + flat_pixels[:,1]

        # HIST EMBEDDINGS
        #embeds = np.zeros((patches.shape[0], embed_dim))
        #for idx in range(len(embeds)):
        #    embed = np.histogram(refactored_patches, bins=np.linspace(0,10,embed_dim+1))[0]
        #    embeds[idx] = embed.astype(np.float)/np.sum(embed)

        embed = np.histogram(refactored_pixels, bins=np.linspace(0, self.step_faktor+1, self.embed_dim+1))[0]
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
        patches = patches.reshape(patches.shape[0], patched_width, patched_width, w, w, 3)
        return patches

    def embed_patches(self, patches, verbose=False):
        patched_xwid = patches.shape[2]
        patched_ywid = patches.shape[1]
        embeds = np.zeros(patches.shape[:3]+(self.embed_dim,))
        for p in range(len(embeds)):
            if verbose:
                print(f"embedding patches  of frame {p} out of {len(embeds)}", end='\r')
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