# %%
import sys
sys.path.append('../../../../')
from continuum_robot.utils import *
# %%
avg_reward_list = load_pickle_file('avg_reward_list')
ep_reward_list = load_pickle_file('scores')

# %%
## Plotting graph
reward_visualization(ep_reward_list, avg_reward_list)
plt.savefig('../../../../docs/images/fixed_goal/reward_pytorch_step_err_comp.png')
plt.show()
# %%
# Plotting graph log scale
reward_log10_visualization(ep_reward_list, avg_reward_list)
plt.savefig('../../../../docs/images/fixed_goal/reward_log_pytorch_step_err_comp.png')
plt.show()
# %%
