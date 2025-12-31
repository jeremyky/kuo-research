from metadrive.policy.base_policy import BasePolicy
from metadrive.envs import MetaDriveEnv
from IPython.display import clear_output, Image

class LeftTurningPolicy(BasePolicy):
    def act(self, agent_id):
        # Always turn left
       # print(agent_id)
        return [0.4, 0.4]
    def act_o(self,agent_id):
        return [0.1,0.5]

env = MetaDriveEnv(dict(agent_policy=LeftTurningPolicy,
                        map="S"))
try:
    env.reset()
    for _ in range(220):
        env.step([-1, -1]) # it doesn't take effect 
        env.render(mode="topdown", 
                   window=False,
                   screen_size=(200, 250),
                   camera_position=(0, 20),
                   screen_record=True)
    env.top_down_renderer.generate_gif()
finally:
    env.close()
    clear_output()
Image(open("demo.gif", 'rb').read())