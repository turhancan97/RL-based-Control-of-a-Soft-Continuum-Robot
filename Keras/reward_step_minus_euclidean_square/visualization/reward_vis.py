# %%
import sys
sys.path.append('../../../')
from continuum_robot.utils import *
# %%
avg_reward_list = load_pickle_file('avg_reward_list')
ep_reward_list = load_pickle_file('ep_reward_list')

# %%
## Plotting graph
reward_visualization(ep_reward_list, avg_reward_list)
plt.savefig('../../../docs/images/random_goal/reward_keras_minus_euc.png')
plt.show()
# %%
# Plotting graph log scale
reward_log10_visualization(ep_reward_list, avg_reward_list)
plt.savefig('../../../docs/images/random_goal/reward_log_keras_minus_eus.png')
plt.show()
# %%
