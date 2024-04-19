import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys

file_name = sys.argv[1]

times = {}

with open(file_name) as file:

    cur_version = ""

    for line in file.readlines():
        elems = line.split()
        
        if len(elems) == 1:
            if not elems[0] in times.keys():
                times[elems[0]] = []
            cur_version = elems[0]
        elif elems[0] == "QED":
            times[cur_version].append(float(elems[2][:-1]))

start_plot_index = 2

medians = np.array([np.median(l) for l in times.values()])[start_plot_index:]
means = np.array([np.mean(l) for l in times.values()])[start_plot_index:]
mins = np.array([np.min(l) for l in times.values()])[start_plot_index:]
maxs = np.array([np.max(l) for l in times.values()])[start_plot_index:]
versions = [key[:] for key in times.keys()][start_plot_index:]

speedup_abs = [1.0]
for time in mins[1:]:
    speedup_abs.append(time / mins[0])
speedup_abs = np.array(speedup_abs)

print(np.max((maxs - mins)/medians))

matplotlib.rcParams.update({'font.size': 10})

fig = plt.figure()
ax = fig.add_subplot(1,1,1)       
ax.barh(versions[::-1], maxs[::-1], label="max")   
ax.barh(versions[::-1], medians[::-1], label="median")
ax.barh(versions[::-1], mins[::-1], label="min", color="g")

for i, rect, sp_abs in zip(range(len(mins)), ax.patches[::-1], speedup_abs):
    if i == 0:
        label = "100%"
        ax.text(rect.get_x() + rect.get_width() + 0.5, rect.get_y() + rect.get_height() / 2, label, ha="center", va="bottom")
    if i >= 1:
        label = "-%d" % (int((1.0 - sp_abs)*100)) + str("%")
        ax.text(rect.get_x() + rect.get_width() + 0.5, rect.get_y() + rect.get_height() / 2, label, ha="center", va="bottom")

#ax.legend()
ax.set_ylabel("version")
ax.set_xlabel("time, s")

ax.grid()
plt.tight_layout()

plt.show()       
        