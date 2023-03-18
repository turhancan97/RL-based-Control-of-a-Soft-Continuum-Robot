import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
############----------------###############
## Adjust the Figure Size at the beginning ##
plt.style.use('ggplot') # ggplot sytle plots
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 25 
plt.rcParams['ytick.labelsize'] = 25 
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["axes.titlesize"] = 'xx-large'
plt.rcParams["axes.labelsize"] = 'xx-large'
# plt.rcParams['animation.ffmpeg_path'] = '/home/tkargin/miniconda3/envs/continuum-rl/bin/ffmpeg' 
## plt.rcParams.keys() ## To see the plot adjustment parameters
############----------------###############

def load_pickle_file(data):
    with open(f'{data}.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data

def reward_visualization(ep_reward_list, avg_reward_list):
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

def reward_log10_visualization(ep_reward_list, avg_reward_list):
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

def plot_various_results(plot_choice,error_store,error_x,error_y,pos_x,pos_y,kappa_1,kappa_2,kappa_3,goal_x,goal_y):
    if plot_choice == 1:
        plt.ylabel("Error",fontsize=20)
        plt.legend(fontsize=15)
        # Error
        plt.plot(range(len(error_store)),error_store,c = 'red',linewidth=2,label='Total Error')
        plt.title("Error Plot of the Test Simulation")
        plt.show()

        # X Error
        plt.plot(range(len(error_x)),error_x,c = 'green',linewidth=2)
        plt.title("Error on the X Axis")
        plt.show()

        # Y Error
        plt.plot(range(len(error_y)),error_y,c = 'blue',linewidth=2)
        plt.title("Error on the Y Axis")
        plt.show()

        # X-Y Error
        plt.plot(range(len(error_x)),error_x,c = 'blue',linewidth=2, label = "X Axis")
        plt.plot(range(len(error_y)),error_y,c = 'green',linewidth=2, label = "Y Axis")
        plt.title("Error on the X-Y Axis")
        plt.show()
    elif plot_choice == 2:
        # X Position
        plt.plot(range(len(pos_x)),pos_x,c = 'green',linewidth=2)
        plt.axhline(y=goal_x)
        plt.legend(["Simulation Result","Reference Signal"])
        plt.title("Trajectory on the X Axis")
        plt.ylabel("X [m]")
        plt.show()

        # Y Position
        plt.plot(range(len(pos_y)),pos_y,c = 'red',linewidth=2)
        plt.axhline(y=goal_y)
        plt.legend(["Simulation Result","Reference Signal"])
        plt.title("Trajectory on the Y Axis")
        plt.ylabel("Y [m]")
        plt.show()

        # X-Y Position
        plt.plot(range(len(pos_x)),pos_x,c = 'red',linewidth=2)
        plt.axhline(y=goal_x)
        plt.plot(range(len(pos_y)),pos_y,c = 'green',linewidth=2)
        plt.axhline(y=goal_y,label='Reference Signal')
        plt.title("Trajectory on the X-Y Axis")
        plt.legend(["X Axis","Reference Signal",'Y Axis','Reference Signal'])
        plt.ylabel("Position [m]",fontsize=20)
        plt.show()
    elif plot_choice == 3:
        plt.legend(fontsize=15)
        # Kappa Plots
        plt.plot(range(len(kappa_1)),kappa_1,c = 'blue',linewidth=2, label = "Curvature-1")
        plt.plot(range(len(kappa_2)),kappa_2,c = 'green',linewidth=2, label = "Curvature-2")
        plt.plot(range(len(kappa_3)),kappa_3,c = 'red',linewidth=2, label = "Curvature-3")
        plt.title("Change of Curvature Values Over Time")
        plt.ylabel(r"Curvature Values $\left [\frac{1}{m}  \right ]$",fontsize=18)
        plt.show()

    plt.xlabel("Step",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(which='major',linewidth=0.7)
    plt.grid(which='minor',linewidth=0.5)
    plt.minorticks_on()
    plt.show()

