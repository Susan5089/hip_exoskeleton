import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
from scipy.interpolate import interp1d

#pos_reward = []
#ee_reward=[]
#cop_left_reward=[]
#cop_right_reward=[]


path1 = "/home/shuzhen/Exo_human_walk_test4_human6_copy2/build/result/"

pathrtss1 = glob(path1+"*pos_reward.txt")
# pathrtss2 = glob(path1+"*cop_left_reward.txt")
pathrtss3 = glob(path1+"*cop_total_reward.txt")
pathrtss4 = glob(path1+"*ee_reward.txt")


cop_left_reward_vector = []
pos_reward_vector =[]
ee_reward_vector=[]
cop_right_reward_vector=[]


for path in pathrtss1:
	pos_reward_ins = np.loadtxt(path)
	# if pos_reward_ins.shape[0]>133:
	pos_reward_ins = pos_reward_ins[-34:]
	pos_reward_vector.append(pos_reward_ins)
	for i in pos_reward_vector:
		print(i.shape)
pos_reward_vector = np.vstack(pos_reward_vector)
pos_reward_mean = pos_reward_vector.mean(axis=0)
pos_reward_std = pos_reward_vector.std(axis=0)


# for path in pathrtss2:
# 	cop_reward_ins = np.loadtxt(path)
# 	# if cop_reward_ins.shape[0]>133:
# 	cop_reward_ins = cop_reward_ins[164:-1]
# 	cop_left_reward_vector.append(cop_reward_ins)
# cop_left_reward_vector = np.vstack(cop_left_reward_vector)
# cop_left_reward_mean = cop_left_reward_vector.mean(axis=0)
# cop_left_reward_std = cop_left_reward_vector.std(axis=0)

x_time = np.loadtxt(path1+"time_vector.txt")
x_time = (x_time[-34:]-15.0)/1.0*100

for path in pathrtss3:
	cop_reward_ins = np.loadtxt(path)
	# if cop_reward_ins.shape[0] > 133:
	cop_reward_ins = cop_reward_ins[-34:]
	cop_right_reward_vector.append(cop_reward_ins)
cop_right_reward_vector = np.vstack(cop_right_reward_vector)
cop_right_reward_mean = cop_right_reward_vector.mean(axis=0)
cop_right_reward_std = cop_right_reward_vector.std(axis=0)



for path in pathrtss4:
	ee_reward_ins = np.loadtxt(path)
	# if ee_reward_ins.shape[0] > 133:
	ee_reward_ins = ee_reward_ins[-34:]
	ee_reward_vector.append(ee_reward_ins)
ee_reward_vector = np.vstack(ee_reward_vector)
ee_reward_mean = ee_reward_vector.mean(axis=0)
ee_reward_std = ee_reward_vector.std(axis=0)
#x_time = np.loadtxt(path1+"t_vector.txt")




# for path in pathrtss3:
# 	cop_right_reward_vector= np.loadtxt(path)
# 	cop_right_reward.append(cop_right_reward_vector.mean())
#
#
#
# for path in pathrtss4:
# 	ee_reward_vector= np.loadtxt(path)
# 	ee_reward.append(ee_reward_vector.mean())
#


plt.figure(0)
plt.xlabel("Gait cycle/(%)", fontsize=13)
plt.ylabel("Rerward", fontsize=13)
#plt.plot(pos_reward, color="blue", linewidth=2.0, linestyle='dotted',label="joint tracking reward")
#plt.plot(y, x0, color="cyan", linewidth=1.5,  label="left foot CoP reward")
#plt.plot(y, x2, color="cyan", linewidth=1.5,  label="left foot CoP reward")
#plt.plot(cop_right_reward, color="red", linewidth=1.5, linestyle="--",label="right foot CoP reward")
#plt.plot(ee_reward, color="magenta", linewidth=1.5, linestyle="-.",label="end effector reward")
plt.plot(x_time, pos_reward_mean, color="blue", linewidth=1.5, linestyle="-", label="joint tracking reward")
# plt.plot(x_time, cop_left_reward_mean, color="cyan", linewidth=1.5,  label="left foot CoP reward")
plt.plot(x_time, cop_right_reward_mean, color="red", linewidth=1.5, linestyle="--", label="foot CoP reward")
plt.plot(x_time, ee_reward_mean, color="magenta", linewidth=1.5, linestyle="-.", label="end effector reward")

plt.fill_between(x_time, pos_reward_mean+pos_reward_std, pos_reward_mean-pos_reward_std, facecolor='blue', alpha=0.1)
# plt.fill_between(x_time, cop_left_reward_mean+cop_left_reward_std, cop_left_reward_mean-cop_left_reward_std, facecolor='cyan', alpha=0.2)
plt.fill_between(x_time, cop_right_reward_mean+cop_right_reward_std, cop_right_reward_mean-cop_right_reward_std, facecolor='red', alpha=0.1)
plt.fill_between(x_time, ee_reward_mean+ee_reward_std, ee_reward_mean-ee_reward_std, facecolor='magenta', alpha=0.1)


plt.legend(loc='best',fontsize=11)
plt.show()

