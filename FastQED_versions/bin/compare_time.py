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

medians = np.array([np.median(l) for l in times.values()])
means = np.array([np.mean(l) for l in times.values()])
mins = np.array([np.min(l) for l in times.values()])
maxs = np.array([np.max(l) for l in times.values()])
versions = [key[14:] for key in times.keys()]

speedup_abs = [1.0]
for time in mins[1:]:
    speedup_abs.append(time / mins[0])
speedup_abs = np.array(speedup_abs)

print(np.max((maxs - mins)/medians))

matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure()
ax = fig.add_subplot(1,1,1)       
#ax.bar(versions, maxs, label="max")   
#ax.bar(versions, medians, label="median")
ax.bar(versions, mins, label="min", color="g")

for i, rect, sp_abs in zip(range(len(mins)), ax.patches, speedup_abs):
    height = rect.get_height()
    if i == 0:
        label = "100%"
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom")
    if i >= 1:
        label = "-%d" % (int((1.0 - sp_abs)*100)) + str("%")
        ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha="center", va="bottom")

#ax.legend()
ax.set_xlabel("version")
ax.set_ylabel("time, s")

ax.grid()

plt.show()       
        