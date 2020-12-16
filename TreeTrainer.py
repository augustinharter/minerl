from .TrainHandler import Handler
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
    discounted = False
    viz = True

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
    delay = 10
    warmup = 20
    gamma = 0.95
    revgamma = 1.1
    datasize = 20000
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
        if True:
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
        else:
            L.exception("Exception occured:"+ str(""))
            print("EXCEPTION")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    