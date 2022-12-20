from gym.envs.registration import register

register(
    id='RLE-v0',
    entry_point='RLE.envs:RLEEnv'
)
