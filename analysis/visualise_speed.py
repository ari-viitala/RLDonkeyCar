import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(record):
    data = pd.read_csv(record, sep=";")
    data["Speed"] = np.sqrt(data.SpeedX**2 + data.SpeedY**2 + data.SpeedZ**2)
    
    return data
    
def plot_reward(data, max_reward=1000):
    
    plt.figure(1, (10, 6))
    plt.title("Episode reward")
    plt.plot(data.groupby("Episode").sum().Reward)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.axhline(max_reward, linestyle="--", color="red", label="Maximum reward")
    plt.legend()
    plt.show()

    
def plot_steps(data):
    
    plt.figure(1, (10, 6))
    plt.title("Steps per episode")
    plt.plot(data.groupby("Episode").max().Step)
    plt.show()
    
    
def plot_speed(data, omit=10, max_speed=0.5):
    
    plt.figure(1, (10, 6))
    plt.title("Episode average speed")
    plt.plot(data[data.Step > omit].groupby("Episode").mean().Speed)
    plt.axhline(max_speed, linestyle="--", color="red", label="Maximum speed")
    plt.legend()
    plt.show()

    
def plot_all(data):

    plt.figure(1, (10, 10))
    for i in range(data["Episode"].max()):
        episode = data[data.Episode == i]
        x = episode["PosX"]
        z = episode["PosZ"]

        plt.plot(x - x[0], z - z[0], color=(i / 100, 0, 0), linewidth=0.5)
        plt.plot(x[-1] - x[0], z[-1] - z[0], marker="x", color="k")

    plt.title("Drift in position over time", size = 14)

    plt.plot(x[-1] - x[0], z[-1] - z[0], marker="x", color="k", label="Termination", linestyle="None")
    plt.legend()
    plt.show()
    
    
def fix_position(data, mean_length=40, line_length=35):

    fixed_x = []
    fixed_z = []

    for e in range(data.Episode.max() + 1):
        
        episode = data[data.Episode == e]
        if len(episode) > 0:
            x = episode["PosX"].values
            z = episode["PosZ"].values

            x = x - x[0]
            z = z - z[0]

            fit = np.polyfit(x[:line_length], z[:line_length], deg=1)

            x_mean = x[:mean_length].mean()

            z_mean = fit[0] * np.sign(x_mean)
            x_mean = np.sign(x_mean)

            angle = np.arccos(x_mean / np.sqrt(x_mean**2 + z_mean**2))

            if x_mean < 0 and fit[0] < 0:
                angle = 2 * np.pi - angle
            elif angle < np.pi / 2 and z[:30].mean() > 0:
                angle = 2 * np.pi - angle

            new_x = x * np.cos(angle) - z * np.sin(angle)
            new_z = x * np.sin(angle) + z * np.cos(angle)

            fixed_x += list(new_x)
            fixed_z += list(new_z)

    data["FixedX"] = fixed_x
    data["FixedZ"] = fixed_z
    
    return data
    
def plot_previous_episodes(data, ax=None, episode=64, heat=False):
    if not ax:
        fig, ax = plt.subplots(1)
        fig.set_size_inches((15, 10))
    for e in range(episode):

        episode = data[data.Episode == e]
        x = episode["FixedX"].values
        z = episode["FixedZ"].values
        s = episode["Speed"].values

        if heat:
            ax.scatter(x, z, c=s, linewidth=0.1, cmap=plt.cm.autumn, s=20, alpha=0.4)
        else:
            ax.plot(x, z, linewidth=0.1, color="k")
            
        if not len(x) == 495:
            ax.plot(x[-1], z[-1], marker="x", color="k", markersize=5)
            
        return ax
            
def plot_episode(data, e):
    episode = data[data.Episode == e]
    x = episode["FixedX"].values
    z = episode["FixedZ"].values

    fig, ax = plt.subplots()
    fig.set_size_inches((15, 10))
    ax.set_ylim(-2.5, 0.5)
    ax.set_xlim(-0.5, 3)
    plot_previous_episodes(data, ax, e, heat=True)
    plt.show(block=False)
    for step in range(0, len(x), 10):
        ax.plot(x[:step], z[:step])
        plt.draw()

            
if __name__ == "__main__":
    data = load_data("../records/log_sac_2020-09-09T16:45:43.295124.csv")
    data = fix_position(data)
    
    for e in range(2):
        plot_episode(data, e)
        
    
    