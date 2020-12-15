import gym
import imageio
import minerl
import pygame
import cv2
import logging
import time
import pickle
import random
from treedetect import TreeDetector
import numpy as np
import torch as T

#from mod.isy_env_wrappers import *
from minerl.viewer.trajectory_display import HumanTrajectoryDisplay

class TreeController():
    def __init__(self, left, right, up, down, attack, jump, unet="", kmeans=""):
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.attack = attack
        self.jump = jump
        self.all = [left,right,up,down,attack,jump]
        self.names = ["left","right","up","down","attack","jump"]
        w = 64
        self.xs = T.arange(w).repeat((w, 1)).float()/w
        self.ys = T.arange(w).repeat((w, 1)).transpose(0, 1).float()/w
        self.lasthighest = 0

        if unet and kmeans:
            self.td = TreeDetector(modelpath=unet, kmeanspath=kmeans)
        else:
            self.td = TreeDetector()

    def navigate(self, obs, verbose=False):
        #print("max obs value", np.max(obs))
        mask = self.td.convert(obs).numpy()

        #find best patch
        number, comps = cv2.connectedComponents((mask>0.2).astype(np.uint8))
        bestidx, highest = 0, 0
        for i in range(1,number):
            value = np.sum(comps==i)
            if value>highest:
                highest = value
                bestidx = i
        patch = comps==bestidx
        x = self.xs[patch].mean()
        y = self.ys[patch].mean()

        lower, upper = 0.45, 0.55
        threshold = 20
        tree = False
        if not (highest<threshold and self.lasthighest<threshold):
            tree = True
            if lower<x<upper:
                # middle
                if lower<y<upper:
                    actions = [self.attack]
                elif y<=lower:
                    actions = [self.up]
                else:
                    actions = [self.down]
            elif x<=lower:
                # left
                if lower<y<upper:
                    actions = [self.left]
                elif y<=lower:
                    actions = [self.left, self.up]
                else:
                    actions = [self.left, self.down]
            else:
                # right
                if lower<y<upper:
                    actions = [self.right]
                elif y<=lower:
                    actions = [self.right, self.up]
                else:
                    actions = [self.right, self.down]
        else:
            actions = [self.jump] 

        if verbose:
            print("tree?", tree, "best idx, and resulting x and y", bestidx, "of", number, x.item(), y.item(), "highest:", highest)
            for act in actions:
                for idx in range(len(self.names)):
                    if self.all[idx].data == act.data:
                        print(self.names[idx])
                
        self.lasthighest = highest
        return tree, actions, mask[:,:,None]


def main():
    with open('./tree-control-stuff/actions_matrix.pkl', 'rb') as f:
        actions_matrix = pickle.load(f)

    logging.basicConfig(level=logging.DEBUG)

    # env = PoVDepthWrapper(env)

    env = gym.make("MineRLTreechopVectorObf-v0")
    env.seed(12)
    obs = env.reset()
    #obs = dict() 
    #obs['pov'] = cv2.cvtColor(cv2.imread("./tree-control-stuff/example.png"), cv2.COLOR_BGR2RGB)

    # # instructions = '{}.render()\n Actions listed below.'.format(
    # #     env.env_spec.name)
    #
    # viewer = HumanTrajectoryDisplay(env.env_spec.name,
    #                                 instructions=instructions, cum_rewards=None)
    # Setup pygame input
    pygame.init()
    display = pygame.display.set_mode((960, 320))
    pygame.event.set_grab(False)

    # Collect images to generate gif
    images = []
    rgb_list = []
    counter = 0
    indx = [[1, 2, 'Left'], [3, 2, 'Right'], [2, 1, 'Up'], [2, 3, 'Down']]

    # SETUP Tree Controller
    left = np.array(actions_matrix[1][2], dtype='float32')
    right = np.array(actions_matrix[3][2], dtype='float32')
    up = np.array(actions_matrix[2][1], dtype='float32')
    down = np.array(actions_matrix[2][3], dtype='float32')
    with open("./tree-control-stuff/attack.p", "rb") as fp:
        attack = pickle.load(fp)
    with open("./tree-control-stuff/act_jf.p", "rb") as fp:
        jump = pickle.load(fp)
    controller = TreeController(left, right, up, down, attack, jump)

    #action = dict()
    action = env.action_space.noop()
    # Loop through the environment
    while True:
        treefound, actions, mask = controller.navigate(obs['pov'][:,:,:3], verbose=True)
        #print(actions)
        if True:
            counter += 1
            # Query input events
            pygame.event.pump()

            # Get keyboard key states
            keys = pygame.key.get_pressed()

            # Initialize using no-operation
            # Moving view with mouse
            # (y, x) = pygame.mouse.get_rel()
            #
            # # Set camera actions
            # action["camera"][0] = x
            # action["camera"][1] = y

            # Toggle mouse grab if ESC pressed
            #if keys[pygame.K_ESCAPE]:
                #pygame.event.set_grab(not pygame.event.get_grab())

            # Move forward action is mapped to W key
            if keys[pygame.K_w]:
                # action["forward"] = 1
                i, j, l = indx[2]
                tmp = np.array(actions_matrix[i][j], dtype='float32')
                actions = [tmp]

            # Move backwards action is mapped to S key
            if keys[pygame.K_s]:
                # action["back"] = 1
                i, j, l = indx[3]
                tmp = np.array(actions_matrix[i][j], dtype='float32')
                actions = [tmp]


            # Move left action is mapped to A key
            if keys[pygame.K_a]:
                # action["left"] = 1
                i, j, l = indx[0]
                tmp = np.array(actions_matrix[i][j], dtype='float32')
                actions = [tmp]

            # Move right action is mapped to D key
            if keys[pygame.K_d]:
                # action["right"] = 1
                i, j, l = indx[1]
                tmp = np.array(actions_matrix[i][j], dtype='float32')
                actions = [tmp]

            # # Jumping mapped to space bar
            # if keys[pygame.K_SPACE]:
            #     action["jump"] = 1
            #
            # # Sneaking mapped to left control key
            # if keys[pygame.K_LCTRL]:
            #     action["sneak"] = 1
            #
            # # Sprinting mapped to left shitf key
            # if keys[pygame.K_LSHIFT]:
            #     action["sprint"] = 1
            #
            # # Attack with mouse click
            # action["attack"] = pygame.mouse.get_pressed()[0]
            #print(action)
        
        # Send action to environment step and receive observation
        for act in actions:
            action['vector'] = act
            obs, rew, done, act = env.step(action)
        rgb = obs['pov'][:,:,:3]
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        #print("gray shape", gray.shape)
        gray = np.stack((gray,gray,gray), axis=-1)
        merge = (rgb*mask + (gray*(1-mask)))
        mask_view = np.concatenate((mask,mask,mask), axis=2)*255

        # DISPLAY UPDATE:
        merge = cv2.resize(merge, (320,320))
        mask_view = cv2.resize(mask_view, (320,320))
        frame = cv2.resize(rgb, (320,320))
        #print(np.max(frame), np.max(mask_view), np.max(merge))
        combined = np.swapaxes(np.concatenate((frame,merge,mask_view), axis=1).astype(np.uint8),0,1)
        surf = pygame.surfarray.make_surface(combined)
        display.blit(surf,(0,0))
        pygame.display.update()
        time.sleep(1/5)


if __name__ == '__main__':
    main()
