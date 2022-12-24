import matplotlib.pyplot as plt
import numpy as np 

time_vector = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/time_vector.txt")[:510]
pos_reward_vector = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/pos_reward_vector.txt")[:510]
ee_reward_vector = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/ee_reward_vector.txt")[:510]
cop_left_reward_vector = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/cop_left_reward_vector.txt")[:510]
cop_right_reward_vector = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/cop_right_reward_vector.txt")[:510]
hip_torque = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/hip_torque.txt")[:510]
knee_torque = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/knee_torque.txt")[:510]
ankle_torque = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/ankle_torque.txt")[:510]
hip_action = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/hip_action.txt")[:510]
knee_action = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/knee_action.txt")[:510]
ankle_action = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/ankle_action.txt")[:510]
hip_force = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/hip_force.txt")[:510]
femur_force = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/femur_force.txt")[:510]
tibia_force = np.loadtxt("/home/shuzhen/Exo_human_squat_test/build/tibia_force.txt")[:510]
plt.figure(0)
plt.subplot(3,1,1)


plt.ylabel("Real-time Reward", fontsize=15)
plt.plot(time_vector,pos_reward_vector, color="black", linewidth=2,  linestyle="--", label="pos_reward")
plt.plot(time_vector,ee_reward_vector, color="magenta", linewidth=2,  linestyle=":", label="ee_reward")
plt.plot(time_vector,cop_left_reward_vector, color="green", linewidth=2,  marker="+", label="cop_left_reward")
plt.plot(time_vector,cop_right_reward_vector, color="red", linewidth=2, label="cop_right_reward")
plt.legend(loc='upper left',fontsize=17);


plt.subplot(3,1,2)

plt.ylabel("Torque/N*m", fontsize=15)
plt.plot(time_vector,hip_torque, color="blue", linewidth=2,  label="hip_torque")
plt.plot(time_vector,knee_torque, color="red", linewidth=2,  linestyle="-.", label="knee_torque")
plt.plot(time_vector,ankle_torque, color="black", linewidth=2, linestyle="--", label="ankle_torque")
plt.legend(loc='upper left',fontsize=17);


plt.subplot(3,1,3)
plt.xlabel("Time/s", fontsize=16)
plt.ylabel("Action from NN/rad", fontsize=15)
plt.plot(time_vector,hip_action, color="blue", linewidth=2,  label="hip_action")
plt.plot(time_vector,knee_action, color="red", linewidth=2,  linestyle="-.", label="knee_action")
plt.plot(time_vector,ankle_action, color="black", linewidth=2, linestyle="--", label="ankle_action")

plt.legend(loc='upper left',fontsize=17);

plt.figure(1)
plt.xlabel("Time/s", fontsize=10)
plt.ylabel("Human perburbation force/N", fontsize=10)
plt.plot(time_vector,hip_force, color="blue", linewidth=2,  label="hip_force")
plt.plot(time_vector,femur_force, color="red", linewidth=2,  linestyle="-.", label="femur_force")
plt.plot(time_vector,tibia_force, color="black", linewidth=2, linestyle="--", label="tibia_force")
plt.legend(loc='upper left',fontsize=13);



plt.show()