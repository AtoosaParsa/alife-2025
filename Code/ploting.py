import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.rcParams['axes.linewidth'] = 1.5

EXP = 8
FREQUENCY = 200
AMPLITUDE = 1.0

dishNum = 1
coordinates1 = []
amplitudes1 = []
with (open(f'EXP/coordinates_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'rb')) as openfile:
    coordinates1 = pickle.load(openfile)
print(f"len(coordinates): {len(coordinates1)}")
with (open(f'EXP/amplitudes_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'rb')) as openfile:
    amplitudes1 = pickle.load(openfile)

dishNum = 2
coordinates2 = []
amplitudes2 = []
with (open(f'EXP/coordinates_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'rb')) as openfile:
    coordinates2 = pickle.load(openfile)
print(f"len(coordinates): {len(coordinates2)}")
with (open(f'EXP/amplitudes_exp{EXP}_dish{dishNum}_f{FREQUENCY}_a{AMPLITUDE}.pickle', 'rb')) as openfile:
    amplitudes2 = pickle.load(openfile)

fig, ax = plt.subplots()
for tid, traj in coordinates1.items():
    #print(f"tid: {tid}")
    #print(coordinates[tid])
    if tid == 1:
        traj = np.array(traj)
        print(len(traj))
        ax.plot(traj[:, 0], traj[:, 1], label=f"Bot {1}", color='royalblue', linewidth=2, linestyle='solid')
ax.set_title(f"Trajectories", fontsize=16)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.set_ylim([0, 1080])
ax.set_xlim([0, 1920])
ax.tick_params(axis='both', which='major', labelsize=14, width=2.5)
ax.tick_params(axis='both', which='minor', labelsize=14, width=1.5)
ax.set_xlabel("X coordinate", fontsize=14)
ax.set_ylabel("Y coordinate", fontsize=14)
legend = ax.legend(loc='upper right', fontsize=14)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))
fig.savefig(f"EXP/trajectories_exp{EXP}_dish{1}_f{FREQUENCY}_a{AMPLITUDE}.png", bbox_inches='tight', format='png', dpi=600, transparent=True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots()
for tid, traj in coordinates2.items():
    #print(f"tid: {tid}")
    #print(coordinates[tid])
    if tid == 1:
        traj = np.array(traj)
        print(len(traj))
        ax.plot(traj[:, 0], traj[:, 1], label=f"Bot {2}", color='royalblue', linewidth=2, linestyle='solid')
ax.set_title(f"Trajectories", fontsize=16)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.set_ylim([0, 1080])
ax.set_xlim([0, 1920])
ax.tick_params(axis='both', which='major', labelsize=14, width=2.5)
ax.tick_params(axis='both', which='minor', labelsize=14, width=1.5)
ax.set_xlabel("X coordinate", fontsize=14)
ax.set_ylabel("Y coordinate", fontsize=14)
legend = ax.legend(loc='upper right', fontsize=14)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))
fig.savefig(f"EXP/trajectories_exp{EXP}_dish{2}_f{FREQUENCY}_a{AMPLITUDE}.png", bbox_inches='tight', format='png', dpi=600, transparent=True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(amplitudes1, color='royalblue', linewidth=2, linestyle='solid')
ax.axhline(y=0.7, xmin=0.0, xmax=len(amplitudes1), color='tomato', linewidth=2, label='Threshold', linestyle='dashed')
ax.set_title(f"Bot 1", fontsize=16)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.set_ylim([-0.1, 1.1])
ax.tick_params(axis='both', which='major', labelsize=14, width=2.5)
ax.tick_params(axis='both', which='minor', labelsize=14, width=1.5)
ax.set_xlabel("Frames (FPS=25)", fontsize=14)
ax.set_ylabel("Straightness Index", fontsize=14)
#legend = ax.legend(loc='upper right', fontsize=14)
#legend.get_frame().set_alpha(None)
#legend.get_frame().set_facecolor((1, 1, 1, 1))
fig.savefig(f"EXP/amplitudes_exp{EXP}_dish{1}_f{FREQUENCY}_a{AMPLITUDE}.png", bbox_inches='tight', format='png', dpi=600, transparent=True)
plt.show()
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(amplitudes2, color='royalblue', linewidth=2, linestyle='solid')
ax.axhline(y=0.7, xmin=0.0, xmax=len(amplitudes2), color='tomato', linewidth=2, label='Threshold', linestyle='dashed')
ax.set_title(f"Bot 2", fontsize=16)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.set_ylim([-0.1, 1.1])
ax.tick_params(axis='both', which='major', labelsize=14, width=2.5)
ax.tick_params(axis='both', which='minor', labelsize=14, width=1.5)
ax.set_xlabel("Frames (FPS=25)", fontsize=14)
ax.set_ylabel("Straightness Index", fontsize=14)
#legend = ax.legend(loc='upper right', fontsize=14)
#legend.get_frame().set_alpha(None)
#legend.get_frame().set_facecolor((1, 1, 1, 1))
fig.savefig(f"EXP/amplitudes_exp{EXP}_dish{2}_f{FREQUENCY}_a{AMPLITUDE}.png", bbox_inches='tight', format='png', dpi=600, transparent=True)
plt.show()
plt.close(fig)