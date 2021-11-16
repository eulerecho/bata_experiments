from mjrl.utils.gym_env import GymEnv
import mjrl.envs
import mj_envs
import time

# env = gym.make("pretouch_mjrl-v0")
env = GymEnv('pretouch_mjrl-v0')
# env = gym.make("Humanoid-v2")
observation = env.reset()


start=time.time()
for _ in range(500):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  #print(action)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()

end=time.time()-start
print(end)
