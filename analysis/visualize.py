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
    
    
    for i, f in enumerate(frames):
        
        start = datetime.datetime.fromisoformat(f["Time"].iloc[0])
        end = datetime.datetime.fromisoformat(f["Time"].iloc[-1])
        
        print("Run {}, Episodes: {}, Time: {:.0f} minutes".format(
            i + 1,
            len(f),
            (end - start).seconds / 60
        ))
    
    plt.figure(1, (10, 6))
    for i in frames:
        #plt.scatter(i["Episode"], i["Reward"], color="k", marker="x", linewidth=0.5)
        plt.plot(i["Episode"], i["Reward"], linewidth=0.5)
    plt.plot(total["Reward"].rolling(10).mean(), label="Average episode reward 10 episode rolling mean", linewidth=2, color = "k")
    plt.axhline(random_mean, linestyle="--", color = "red", label="Random episode average reward")
    
    if real_car:
        plt.axhline(150, linestyle="--", color="b", label="~ One lap")
    
    plt.ylim(0)
    plt.title("Reward per episode")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.show()
    
    plt.figure(1, (10, 6))
    for i in frames:
        plt.scatter(i["Reward"].cumsum(), i["Reward"], linewidth=0.7, marker="x")
    plt.plot(total["Reward"].cumsum(), total["Reward"].rolling(10).mean(), label="Average episode reward", linewidth=2, color = "k")
    plt.axhline(random_mean, linestyle="--", color = "red", label="Random episode average reward")
    
    if real_car:
        plt.axhline(150, linestyle="--", color="b", label="~ One lap")
    
    plt.ylim(0)
    plt.legend()
    plt.xlabel("Cumulative environment steps")
    plt.ylabel("Episode reward")
    
    plt.show()