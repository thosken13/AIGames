import gym

env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)
for episode in range(10):
    observation = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break