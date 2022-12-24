from Model import SimulationNN
import torch
from scipy.io import loadmat
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


model = SimulationNN(num_states = 18,num_actions=2)


ckpt = "/home/shuzhen/Downloads/Hip_exoskeleton_NCSU22_old42/nn/new_new2/11.pt"


model.load_state_dict(torch.load(ckpt))
model.eval()

path ="/home/shuzhen/Downloads/Shuzhen1_1_0.mat"

IMU_data = loadmat(path)

# print(IMU_data)
# print(type(IMU_data))


path ="/home/shuzhen/Downloads/Hip_exoskeleton_NCSU22_old42/build/"


y_bspl=[]
x_bspl=[]
with open(path+"hip_r_exo_angle.txt","r") as f:
	data0 = f.readlines()
data0 = [float(i.rstrip()) for i in data0]
data = data0
data = np.array(data)*np.pi/180


qT_L = IMU_data["qT_L"][30000:33000]*np.pi/180
qT_R = IMU_data["qT_R"][30000:33000]*np.pi/180
dqT_L = IMU_data["dqT_L"][30000:33000]*np.pi/180
dqT_R = IMU_data["dqT_R"][30000:33000]*np.pi/180

torque_l = IMU_data["Motor_Torque_Reference_Left"][30000:33000]
torque_r = IMU_data["Motor_Torque_Reference_Right"][30000:33000]



qT_R = qT_R[np.array(list(range(0,len(qT_R),5)))]
qT_L = qT_L[np.array(list(range(0,len(qT_L),5)))]
dqT_L = dqT_L[np.array(list(range(0,len(dqT_L),5)))]
dqT_R = dqT_R[np.array(list(range(0,len(dqT_R),5)))]

# plt.plot(qT_R,color="red", linewidth=2)
# plt.plot(data,color="cyan", linewidth=2)


# plt.show()


# exit()

# intialize input 
pv_history = deque(maxlen=12)
a_history = deque(maxlen=6)
for _ in range(12):
    pv_history.append(0)
for _ in range(6):
    a_history.append(0)

action_res = []

for i in range(len(qT_R)):
    # print("loop", np.array(pv_history, np.float).shape)
    # print("loop", np.array(a_history, np.float).shape)
    input_np = np.concatenate((np.array(pv_history, np.float).flatten(), np.array(a_history, np.float)))
    
    input_tensor = torch.from_numpy(input_np).float()
    pred_act = model.get_action(input_tensor)
    # update buffer
    pv_history.append(qT_L[i])
    pv_history.append(qT_R[i])
    pv_history.append(dqT_L[i])
    pv_history.append(dqT_R[i])
    a_history.append(pred_act[0])
    a_history.append(pred_act[1])
    
    action_res.append(pred_act)

action_res = np.stack(action_res, axis=0)
print(action_res[:,1].shape)

fig = plt.figure()
# plt.plot(0.1*action_res[:,0],color="blue", linewidth=2)
# plt.plot(0.1*action_res[:,1],color="red", linewidth=2)

plt.plot(action_res[:,0],color="red", linewidth=2)
# plt.plot(torque_l,color="blue", linewidth=2)
# plt.plot(qT_R,color="cyan", linewidth=2)

plt.show()