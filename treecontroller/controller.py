import gym
import imageio
import minerl
import pygame
import cv2
import logging
import time
import pickle
import random

#from mod.isy_env_wrappers import *
from minerl.viewer.trajectory_display import HumanTrajectoryDisplay


def main():
    with open('actions_matrix.pkl', 'rb') as f:
        actions_matrix = pickle.load(f)

    logging.basicConfig(level=logging.DEBUG)

    env = gym.make("MineRLTreechopVectorObf-v0")
    # env = PoVDepthWrapper(env)

    env.seed(42)
    obs = env.reset()

    # # instructions = '{}.render()\n Actions listed below.'.format(
    # #     env.env_spec.name)
    #
    # viewer = HumanTrajectoryDisplay(env.env_spec.name,
    #                                 instructions=instructions, cum_rewards=None)
    # Setup pygame input
    pygame.init()
    pygame.display.set_mode((640, 640))
    pygame.event.set_grab(False)

    # Collect images to generate gif
    images = []
    rgb_list = []
    counter = 0
    indx = [[1, 2, 'Left'], [3, 2, 'Right'], [2, 1, 'Up'], [2, 3, 'Down']]
    action = env.action_space.noop()
    # Loop through the environment
    while True:
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
            action['vector'] = np.array(actions_matrix[i][j], dtype='float32')

        # Move backwards action is mapped to S key
        if keys[pygame.K_s]:
            # action["back"] = 1
            i, j, l = indx[3]
            action['vector'] = np.array(actions_matrix[i][j], dtype='float32')


        # Move left action is mapped to A key
        if keys[pygame.K_a]:
            # action["left"] = 1
            i, j, l = indx[0]
            action['vector'] = np.array(actions_matrix[i][j], dtype='float32')

        # Move right action is mapped to D key
        if keys[pygame.K_d]:
            # action["right"] = 1
            i, j, l = indx[1]
            action['vector'] = np.array(actions_matrix[i][j], dtype='float32')

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
        print(action)
        # Send action to environment step and receive observation
        obs, rew, done, act = env.step(action)

        rgb = obs['pov'][:,:,:3]
        # depth = obs['pov'][:,:,-1]

        obs['pov'] = rgb
        # View the observation
        # viewer.render(obs, rew, done, action, 0, 1)

        # Append observation + segmentation to image collection
        # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        # images.append(np.hstack([rgb,depth]))
        # rgb_list.append(rgb)
        # imageio.imwrite('imgs/' + str(counter).zfill(4) + '.png', rgb)

        # Use G key to save image list as gif
        # if keys[pygame.K_g]:
        #     imageio.mimsave(f'keyboard-{counter}.gif', images)
        #     counter += 1
        #     images = []
        #
        # if keys[pygame.K_h] and len(images) >= 100:
        #     for idx, elem in enumerate(rgb_list):
        #         imageio.imwrite(f"/home/dennis/Schreibtisch/traj/{idx}.png", elem)
        #     #imageio.mimwrite("/home/dennis/Schreibtisch/traj/", rgb_list)
        #
        time.sleep(1/5)


if __name__ == '__main__':
    main()
