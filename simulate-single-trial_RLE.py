import time
import numpy as np
import pandas as pd
from scipy.special import softmax
import re
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import RLE
from ray.tune.registry import register_env
from RLE.envs.RLEEnv import RLEEnv

ENV_NAME = 'RLE-v0'
register_env(ENV_NAME, lambda config: RLEEnv(config))
ray.init(ignore_reinit_error=True, log_to_driver=False)

scenarioID = 8
SAFE_MODE = True
checkpoint_path = 'checkpoint/checkpoint_003000'
measure_names = ['MTD', 'reward']
state_names = ['state' + str(num).zfill(2) for num in range(1+6+6+2)]
prob_names = ['prob' + str(i) for i in range(3+6)]

config = DEFAULT_CONFIG.copy()
config['seed'] = 1234
config['gamma'] = 1.0
config['framework'] = 'torch'
config['num_workers'] = 1
config['num_sgd_iter'] = 20
config['num_cpus_per_worker'] = 1
config['sgd_minibatch_size'] = 200
config['train_batch_size'] = 10000
sim_config = {'D':6, 'N_cohort':3, 'N_total':36, 'phi':0.25, 'scenario':str(scenarioID)}
config['env_config'] = sim_config
agent = PPOTrainer(config, ENV_NAME)
env = gym.make(ENV_NAME, config = sim_config)
agent.load_checkpoint(checkpoint_path)

results_score = []
results_cohort = []

simID = 123
env.seed(simID)
state = env.reset()
done = False
cohortID = 1
while not done:
    action = agent.compute_single_action(state, full_fetch = True)
    probs = softmax(action[2]['action_dist_inputs'])
    action = np.argmax(probs)

    if SAFE_MODE:
        current_dose = (np.rint(state[0] * (sim_config['D']-1))).astype(np.int)
        Ns = np.round(state[1:(sim_config['D']+1)] * sim_config['N_total'])
        DLTs = np.round(state[(sim_config['D']+1):(2*sim_config['D']+1)] * sim_config['N_total'])
        ratios = DLTs / Ns
        if ratios[current_dose] > sim_config['phi'] and current_dose <= sim_config['D'] - 2 and action == 2:
            action = 1
        elif ratios[current_dose] > 2*sim_config['phi'] and current_dose >= 1 and 1 <= action <= 2:
            action = 0
        elif ratios[current_dose] < sim_config['phi'] and action == 0:
            action = 1

    results_cohort.append([scenarioID, simID, cohortID, *probs, action, *state])
    state, reward, done, info = env.step(action)
    cohortID += 1
    if done:
        measures = [info['MTD'], reward]
        results_score.append([scenarioID, simID, *measures, *state])

df_score = pd.DataFrame(results_score, columns=['scenarioID', 'simID', *measure_names, *state_names])
df_cohort = pd.DataFrame(results_cohort, columns=['scenarioID', 'simID', 'cohortID', *prob_names, 'action', *state_names])
df_score.to_csv('evaluation_score.csv', index=False)
df_cohort.to_csv('evaluation_cohort.csv', index=False)

ray.shutdown()
