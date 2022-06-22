import os
import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import *


def render_level(layout_name):
    _, _, env = setup_env(layout_name)
    print("size: ", (env.mdp.width, env.mdp.height))
    env.render()
    time.sleep(60)


def setup_env(layout_name):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
    agent1 = RandomAgent(all_actions=True)
    agent2 = RandomAgent(all_actions=True)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    agent1.set_mdp(mdp)
    agent2.set_mdp(mdp)
    return agent1, agent2, env


agent1, agent2, env = setup_env("train_gan_small/gen2_basic_6-6-4")
done = False
while not done:
    env.render()
    joint_action = (agent1.action(env.state)[0], agent2.action(env.state)[0])
    next_state, timestep_sparse_reward, done, info = env.step(joint_action)
    time.sleep(0.1)