import gym

#env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
env.reset()
print(env.action_space)
print(env.observation_space)
done = False
i=0
while not done:
    if i % 10 ==0:
        env.render()
    action = 2#env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(action, obs, reward, done)
    i+=1
   # env.render()