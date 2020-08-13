import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

def visualize_run(folder, episodes=None, real_car=False):
    
    frames = [pd.read_csv(folder + x, sep=";") for x in os.listdir(folder)]
    frames = [f for f in frames if len(f) > 0]
    
    if episodes:
        frames = [f for f in frames if len(f) == episodes]
        
    data = pd.concat(frames)
    total = data.groupby("Episode").mean().reset_index()
    random_mean = total[total["Episode"] < 5].mean()["Reward"]
    
    
    if real_car:
        for i, f in enumerate(frames):

            start = datetime.datetime.fromisoformat(f["Time"].iloc[0])
            end = datetime.datetime.fromisoformat(f["Time"].iloc[-1])

            print("Run {}, Episodes: {}, Time: {:.0f} minutes".format(
                i + 1,
                len(f),
                (end - start).seconds / 60
            ))
    
    plt.figure(1, (20, 6))
    plt.subplot(1, 2, 1)
    #plt.figure(1, (10, 6))
    for i in frames:
        #plt.scatter(i["Episode"], i["Reward"], color="k", marker="x", linewidth=0.5)
        plt.plot(i["Episode"], i["Reward"], linewidth=0.5)
    plt.plot(total["Reward"].rolling(10).mean(), label="Average episode reward", linewidth=2, color = "k")
    plt.axhline(random_mean, linestyle="--", color = "red", label="Random policy")
    
    if real_car:
        plt.axhline(150, linestyle="--", color="b", label="~ One lap")
    
    plt.ylim(0)
    plt.title("Reward per episode")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    #plt.show()
    
    plt.subplot(1, 2, 2)
    #plt.figure(1, (10, 6))
    for i in frames:
        plt.plot(i["Reward"].cumsum(), i["Reward"], linewidth=0.7)
    #plt.plot(total["Reward"].cumsum(), total["Reward"].rolling(10).mean(), label="Average episode reward", linewidth=2, color = "k")
    plt.axhline(random_mean, linestyle="--", color = "red", label="Random episode average reward")
    
    if real_car:
        plt.axhline(150, linestyle="--", color="b", label="~ One lap")
    
    plt.ylim(0)
    plt.legend()
    plt.xlabel("Cumulative environment steps")
    plt.ylabel("Episode reward")
    
    plt.show()
    
def visualize_ewm(folder, alpha=0.05):
    frames = [pd.read_csv(folder + x, sep=";") for x in os.listdir(folder)]
    frames = [f for f in frames if len(f) > 0]
    rewards = [f["Reward"] for f in frames if len(f) > 0]

    steprewards = pd.concat([pd.DataFrame({"step": f["Reward"].cumsum(), "reward": f["Reward"]}) for f in frames]).sort_values("step")

    ewm = steprewards["reward"].ewm(alpha=alpha).mean()
    std = steprewards["reward"].ewm(alpha=alpha).std()

    plt.figure(1, (10, 6))
    plt.fill_between(steprewards["step"], ewm - std, ewm + std, alpha=0.2, color="k", label="Standard deviation")
    plt.scatter(steprewards["step"], steprewards["reward"], marker="x", alpha=0.4, label="Individual episodes")
    plt.plot(steprewards["step"], ewm, label="EWMA alpha=0.05", linewidth=2, color="red")
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward")
    plt.legend(loc="upper left")

    plt.show()
    
def visualize_ewm_time(folder, alpha=0.05):
    frames = [pd.read_csv(folder + x, sep=";") for x in os.listdir(folder)]
    frames = [f for f in frames if len(f) > 0]
    
    rewards = [f["Reward"] for f in frames if len(f) > 0]

    steprewards = pd.concat([pd.DataFrame({"time": (pd.to_datetime(f["Time"]) - pd.to_datetime(f["Time"])[0]).apply(lambda x: x.seconds) / 60, "reward": f["Reward"]}) for f in frames]).sort_values("time")

    ewm = steprewards["reward"].ewm(alpha=alpha).mean()
    std = steprewards["reward"].ewm(alpha=alpha).std()

    
    fig, ax = plt.subplots(constrained_layout=True)
    plt.figure(1, (10, 6))
    plt.fill_between(steprewards["time"], ewm - std, ewm + std, alpha=0.2, color="k", label="Standard deviation")
    plt.scatter(steprewards["time"], steprewards["reward"], marker="x", alpha=0.4, label="Individual episodes")
    plt.plot(steprewards["time"], ewm, label="EWMA alpha=0.05", linewidth=2, color="red")
    plt.xlabel("Training time (minutes)")
    plt.ylabel("Episode reward")
    plt.legend(loc="upper left")

    plt.show()
    
def visualize_ewm_both(folder, alpha):
    
    frames = [pd.read_csv(folder + x, sep=";") for x in os.listdir(folder)]
    frames = [f for f in frames if len(f) > 0]

    rewards = [f["Reward"] for f in frames if len(f) > 0]

    steprewards = pd.concat([pd.DataFrame({"step": f["Reward"].cumsum(), "reward": f["Reward"]}) for f in frames]).sort_values("step")

    ewm = steprewards["reward"].ewm(alpha=alpha).mean()
    std = steprewards["reward"].ewm(alpha=alpha).std()



    fig, ax = plt.subplots(constrained_layout=True)
    fig.set_size_inches((10, 5))
    ax.fill_between(steprewards["step"], ewm - std, ewm + std, alpha=0.2, color="k", label="Standard deviation")
    ax.scatter(steprewards["step"], steprewards["reward"], marker="x", alpha=0.4, label="Individual episodes")
    ax.plot(steprewards["step"], ewm, label="Mean", linewidth=2, color="red")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.legend(loc="upper left")

    secax = ax.secondary_xaxis("top", functions=(lambda x: x / 600, lambda x: x * 600))
    secax.set_xlabel("Time (min)")

    #plt.show()