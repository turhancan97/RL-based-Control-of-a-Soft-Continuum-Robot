# %%
import sys

sys.path.append("../../")

from continuum_robot.utils import *

import yaml

# load config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
# %%
avg_reward_list = load_pickle_file(
    f'../{config["goal_type"]}/{config["reward_type"]}/rewards/avg_reward_list'
)
ep_reward_list = load_pickle_file(
    f'../{config["goal_type"]}/{config["reward_type"]}/rewards/ep_reward_list'
)

# %%
## Plotting graph
reward_visualization(ep_reward_list, avg_reward_list)
# plt.savefig(
#     f'../../docs/images/{config["goal_type"]}/reward_keras_{config["reward_type"]}.png'
# )
plt.show()
# %%
# Plotting graph log scale
reward_log10_visualization(ep_reward_list, avg_reward_list)
# plt.savefig(
#     f'../../docs/images/{config["goal_type"]}/reward_log_keras_{config["reward_type"]}.png'
# )
plt.show()
