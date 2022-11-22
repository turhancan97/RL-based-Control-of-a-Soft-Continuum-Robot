# %%
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# %%
def load_pickle_file(data):
    with open(f'{data}.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data

avg_reward_list = load_pickle_file('avg_reward_list')
ep_reward_list = load_pickle_file('ep_reward_list')

# %%
## Plotting graph
plt.rcParams['xtick.labelsize'] = 25 
plt.rcParams['ytick.labelsize'] = 25 
fig, axs = plt.subplots(1, 2,figsize=(26, 13))

# Episodes versus Episodic Rewards
# Give plot a gray background like ggplot.
axs[0].set_facecolor('#EBEBEB')

# Remove border around plot.
[axs[0].spines[side].set_visible(False) for side in axs[0].spines]

axs[0].plot(np.arange(1, len(ep_reward_list)+1), ep_reward_list,'black',alpha=.9)
axs[0].set_xlabel('\nEpisode Number',fontsize=35)
axs[0].set_ylabel('Reward',fontsize=35)
axs[0].set_title('Episodic Reward',fontsize=35,fontweight="bold",fontname="Times New Roman")

# Style the grid.
axs[0].grid(which='major', color='white', linewidth=1.2)
axs[0].grid(which='minor', color='white', linewidth=0.6)
# Show the minor ticks and grid.
axs[0].minorticks_on()
# Now hide the minor ticks (but leave the gridlines).
axs[0].tick_params(which='minor', bottom=False, left=False)


# Episodes versus Avg. Rewards
# Give plot a gray background like ggplot.
axs[1].set_facecolor('#EBEBEB')

# Remove border around plot.
[axs[1].spines[side].set_visible(False) for side in axs[0].spines]

axs[1].plot(np.arange(1, len(avg_reward_list)+1), avg_reward_list,'black',linewidth=4)
axs[1].set_xlabel('\nEpisode Number',fontsize=35)
axs[1].set_ylabel('Reward',fontsize=35)
axs[1].set_title('Average Episodic Reward',fontname="Times New Roman",fontsize=35,fontweight="bold")

# Style the grid.
axs[1].grid(which='major', color='white', linewidth=1.2)
axs[1].grid(which='minor', color='white', linewidth=0.6)
# Show the minor ticks and grid.
axs[1].minorticks_on()
# Now hide the minor ticks (but leave the gridlines).
axs[1].tick_params(which='minor', bottom=False, left=False)

# Only show minor gridlines once in between major gridlines.
axs[0].xaxis.set_minor_locator(AutoMinorLocator(2))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
axs[1].xaxis.set_minor_locator(AutoMinorLocator(2))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))

plt.savefig('../../docs/images/reward_keras.png')
plt.show()
# %%
# Plotting graph log scale
fig, axs = plt.subplots(1, 2,figsize=(20, 10))

# Episodes versus Episodic Rewards
# Give plot a gray background like ggplot.
axs[0].set_facecolor('#EBEBEB')

# Remove border around plot.
[axs[0].spines[side].set_visible(False) for side in axs[0].spines]

axs[0].plot(np.log10(np.arange(1, len(ep_reward_list)+1)), ep_reward_list,'black',alpha=.9)
axs[0].set_xlabel('Episode Number (log10 Scale)',fontsize=25)
axs[0].set_ylabel('Reward',fontsize=15)
axs[0].set_title('Episodic Reward',fontsize=25,fontweight="bold",fontname="Times New Roman")

# Style the grid.
axs[0].grid(which='major', color='white', linewidth=1.2)
axs[0].grid(which='minor', color='white', linewidth=0.6)
# Show the minor ticks and grid.
axs[0].minorticks_on()
# Now hide the minor ticks (but leave the gridlines).
axs[0].tick_params(which='minor', bottom=False, left=False)


# Episodes versus Avg. Rewards
# Give plot a gray background like ggplot.
axs[1].set_facecolor('#EBEBEB')

# Remove border around plot.
[axs[1].spines[side].set_visible(False) for side in axs[0].spines]

axs[1].plot(np.log10(np.arange(1, len(avg_reward_list)+1)), avg_reward_list,'black',linewidth=4)
axs[1].set_xlabel('Episode Number (log10 Scale)',fontsize=25)
axs[1].set_ylabel('Reward',fontsize=15)
axs[1].set_title('Average Episodic Reward',fontname="Times New Roman",fontsize=25,fontweight="bold")

# Style the grid.
axs[1].grid(which='major', color='white', linewidth=1.2)
axs[1].grid(which='minor', color='white', linewidth=0.6)
# Show the minor ticks and grid.
axs[1].minorticks_on()
# Now hide the minor ticks (but leave the gridlines).
axs[1].tick_params(which='minor', bottom=False, left=False)

# Only show minor gridlines once in between major gridlines.
axs[0].xaxis.set_minor_locator(AutoMinorLocator(2))
axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
axs[1].xaxis.set_minor_locator(AutoMinorLocator(2))
axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))

plt.savefig('../../docs/images/reward_log_keras.png')
plt.show()
# %%