import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

simple_rows = []
rtg_rows = []

# read the simple policy gradient results
with open("simple_pg_results.txt", "r") as file:
    for line in file:
        # Split by comma, strip spaces, and convert to int (or float)
        line_list = line.strip().split(",")
        line_list.pop()
        row = [float(x.strip()) for x in line_list]
        row.pop()
        simple_rows.append(np.array(row))

# read the reward to go policy gradient results
with open("rtg_results.txt", "r") as file:
    for line in file:
        # Split by comma, strip spaces, and convert to int (or float)
        line_list = line.strip().split(",")
        line_list.pop()
        row = [float(x.strip()) for x in line_list]
        row.pop()
        rtg_rows.append(np.array(row))

simple_avg = np.mean(np.array(simple_rows), axis=0)
rtg_avg = np.mean(np.array(rtg_rows), axis=0)


# plt.scatter(range(len(simple_avg)), simple_avg)
# plt.xlabel("Timesteps")
# plt.ylabel("Average Return Over 5 Runs")
# plt.title("Simple Policy Gradient Performance")
# plt.show()

# plt.scatter(range(len(rtg_avg)), rtg_avg)
# plt.xlabel("Timesteps")
# plt.ylabel("Average Return Over 5 Runs")
# plt.title("Reward To Go Policy Gradient Performance")
# plt.show()

plt.scatter(range(len(simple_avg)), simple_avg, label="Simple PG")
plt.scatter(range(len(rtg_avg)), rtg_avg, label="Reward-to-Go PG")

plt.xlabel("Timesteps")
plt.ylabel("Average Return Over 5 Runs")
plt.title("Policy Gradient Performance Comparison")
plt.legend()  # show the labels
plt.show()
