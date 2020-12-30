from PatchEmbedder import PatchEmbedder
import numpy as np
from sklearn.mixture import GaussianMixture as GMM


if __name__ == '__main__':
    embedder = PatchEmbedder()
    base = np.random.randn(*(10, 64, 64, 3))
    flat_base = base.reshape(-1,3)
    gmm = GMM(n_components=64)

    gmm.fit(flat_base)

    batch = np.random.randn(10*29*29*8*8, 3)
    smartbatch = np.random.randn(10*64*64, 3)
    frame = np.random.randn(8*8*29*29, 3)
    patch = np.random.randn(8*8, 3)

    def predict_batch():
        gmm.predict(batch)

    def predict_frame():
        for _ in range(10):
            gmm.predict(frame)

    def predict_patches():
        for _ in range(10*29*29):
            gmm.predict(patch)

    def predict_batch_smart():
        embedder.predict_batch(smartbatch)

    predict_patches()
    predict_frame()
    predict_batch()
    predict_batch_smart()
