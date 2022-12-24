import time
import numpy as np
import pandas as pd
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import RLE

from ray.tune.registry import register_env
from RLE.envs.RLEEnv import RLEEnv

ENV_NAME = 'RLE-v0'
register_env(ENV_NAME, lambda config: RLEEnv(config))

ray.init(ignore_reinit_error=True, log_to_driver=False)

config = DEFAULT_CONFIG.copy()
config['seed'] = 123
config['gamma'] = 1.0
config['framework'] = 'torch'
config['num_workers'] = 4
config['num_sgd_iter'] = 20
config['num_cpus_per_worker'] = 1
config['sgd_minibatch_size'] = 200
config['train_batch_size'] = 10000
config['env_config'] = {'D':6, 'N_cohort':3, 'N_total':36, 'scenario':'random'}

agent = PPOTrainer(config, ENV_NAME)

N = 3000
results = []
episode_data = []
start_time = time.time()

for n in range(1, N+1):
    result = agent.train()
    results.append(result)
    episode = {'n': n,
               'episode_reward_min':  result['episode_reward_min'],
               'episode_reward_mean': result['episode_reward_mean'],
               'episode_reward_max':  result['episode_reward_max'],
               'episode_len_mean':    result['episode_len_mean']}
    episode_data.append(episode)
    print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
    if n >= 1000 and n % 500 == 0:
        checkpoint_path = agent.save()
        print(checkpoint_path)

end_time = time.time()
print('time spent: ' + str(end_time - start_time))

df = pd.DataFrame(data=episode_data)
df.to_csv('result_learn_RLE.csv', index=False)

ray.shutdown()
