from gym.envs.registration import register

try:
    register(
        id='CustomCartPole-v0',
        entry_point='custom_openai_frameworks.envs:CartPoleEnv',
        max_episode_steps=800
    )
except:
    print("Unable to register custom env")

try:
    register(
        id='CountUp-v0',
        entry_point='custom_openai_frameworks.envs:CountUpEnv',
        max_episode_steps=800
    )
except:
    print("Unable to register custom env")

try:
    register(
        id='GolfCardGame-v0',
        entry_point='custom_openai_frameworks.envs:GolfCardGameEnv',
        max_episode_steps=800
    )
except:
    print("Unable to register custom env")
