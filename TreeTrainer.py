from TrainHandler import Handler
import logging as L
import os

class Arguments():
    train = True
    final = True
    blurr = False
    test = True
    unet = True
    live = True
    critic = True
    distnorm = False
    uload = False
    cload = False
    dream = False
    sigmoid = False
    clustersave = False
    staticL1 = False
    savekmeans = True
    discounted = True
    resnet = False
    viz = True
    grid = False
    dummy = False
    clippify = False
    final = True
    integrated = True
    grounded = False

    embed_dim = 64
    embed_cluster = 64
    embed_train_samples = 100
    datasize = 100
    blur = 0
    delta = 50
    cluster = "5"
    clustercritic = 0
    gray = True
    epochs = 10
    dreamsteps = 0
    threshold = 0.9
    L1 = 20
    L2 = 0
    saveevery = 10
    rewidx = 3
    wait = 120
    delay = 0
    warmup = 20
    gamma = 0.95
    revgamma = 1.1
    trajsize = 50
    testsize = 300
    workers = (4,16,4)
    chunksize = 20
    cons = 350
    color = "HSV"
    name = "final-L20"

class Trainer():
    def train(self):
        args = Arguments()
        #print(args.name)
        H = Handler(args)
        H.load_data()
        H.create_patch_embedding_clusters()
        H.vis_embed()

        # old
        if False:
            if args.cluster:
                if args.train or args.savekmeans:
                    H.cluster(mode="train")
                #if args.test:
                #    H.cluster(mode="test")
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

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    