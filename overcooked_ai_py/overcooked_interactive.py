import pygame
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from argparse import ArgumentParser

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

valid_counters = [(5, 3)]
one_counter_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': valid_counters,
    'counter_drop': valid_counters,
    'counter_pickup': [],
    'same_motion_goals': True
}


class App:
    """Class to run an Overcooked Gridworld game, leaving one of the players as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""
    def __init__(self, env, agent, player_idx, slow_time):
        self._running = True
        self._display_surf = None
        self.env = env
        self.agent = agent
        self.agent_idx = player_idx
        self.slow_time = slow_time
        print("Human player index:", player_idx)

    def on_init(self):
        pygame.init()

        # Adding pre-trained agent as teammate
        self.agent.set_agent_index(self.agent_idx)
        self.agent.set_mdp(self.env.mdp)

        print(self.env)
        self.env.render()
        self._running = True

    def on_event(self, event):
        done = False

        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == pygame.K_UP:
                action = Direction.NORTH
            elif pressed_key == pygame.K_RIGHT:
                action = Direction.EAST
            elif pressed_key == pygame.K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == pygame.K_LEFT:
                action = Direction.WEST
            elif pressed_key == pygame.K_SPACE:
                action = Action.INTERACT

            if action in Action.ALL_ACTIONS:

                done = self.step_env(action)

                if self.slow_time and not done:
                    for _ in range(2):
                        action = Action.STAY
                        done = self.step_env(action)
                        if done:
                            break

        if event.type == pygame.QUIT or done:
            print("TOT rew", self.env.cumulative_sparse_rewards)
            self._running = False

    def step_env(self, my_action):
        agent_action = self.agent.action(self.env.state)[0]

        if self.agent_idx == 0:
            joint_action = (agent_action, my_action)
        else:
            joint_action = (my_action, agent_action)

        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action)

        print(self.env)
        print(joint_action)
        print(self.env.state)
        self.env.render()
        print("Curr reward: (sparse)", timestep_sparse_reward, "\t(dense)",
              info["shaped_r_by_agent"])
        print(self.env.t)
        return done

    def on_loop(self):
        pass

    def on_render(self):
        pass

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    mdp = OvercookedGridworld.from_layout_name("counter_circuit")
    env = OvercookedEnv.from_mdp(mdp)
    rand_agent = RandomAgent(all_actions=True)
    theApp = App(env, rand_agent, player_idx=0, slow_time=True)
    theApp.on_execute()