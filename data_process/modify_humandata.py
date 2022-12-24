import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
from scipy.interpolate import interp1d

#pos_reward = []
#ee_reward=[]
#cop_left_reward=[]
#cop_right_reward=[]


path = "/home/shuzhen/Hip_exoskeleton_NCSU/data/humandata/"


time_ins = []
time_vector = []
PelvisX_vector =[]
PelvisX_ins = []
PelvisY_vector=[]
PelvisZ_vector=[]


time_vector = np.loadtxt(path+"time.txt")[0:373]

R_PelvisX_vector = np.loadtxt(path+"R_PelvisAngleX.txt")[0:373]
R_PelvisY_vector = np.loadtxt(path+"R_PelvisAngleY.txt")[0:373]
R_PelvisZ_vector = np.loadtxt(path+"R_PelvisAngleZ.txt")[0:373]


R_HipX_vector = np.loadtxt(path+"R_HipAngleX.txt")[0:373]
R_HipY_vector = np.loadtxt(path+"R_HipAngleY.txt")[0:373]
R_HipZ_vector = np.loadtxt(path+"R_HipAngleZ.txt")[0:373]


R_kneeX_vector = np.loadtxt(path+"R_KneeAngleX.txt")[0:373]
R_kneeY_vector = np.loadtxt(path+"R_KneeAngleY.txt")[0:373]
R_kneeZ_vector = np.loadtxt(path+"R_KneeAngleZ.txt")[0:373]


R_AnkleX_vector = np.loadtxt(path+"R_AnkleAngleX.txt")[0:373]
R_AnkleY_vector = np.loadtxt(path+"R_AnkleAngleY.txt")[0:373]
R_AnkleZ_vector = np.loadtxt(path+"R_AnkleAngleZ.txt")[0:373]


L_PelvisX_vector = np.loadtxt(path+"L_PelvisAngleX.txt")[0:373]
L_PelvisY_vector = np.loadtxt(path+"L_PelvisAngleY.txt")[0:373]
L_PelvisZ_vector = np.loadtxt(path+"L_PelvisAngleZ.txt")[0:373]


L_HipX_vector = np.loadtxt(path+"L_HipAngleX.txt")[0:373]
L_HipY_vector = np.loadtxt(path+"L_HipAngleY.txt")[0:373]
L_HipZ_vector = np.loadtxt(path+"L_HipAngleZ.txt")[0:373]


L_kneeX_vector = np.loadtxt(path+"L_KneeAngleX.txt")[0:373]
L_kneeY_vector = np.loadtxt(path+"L_KneeAngleY.txt")[0:373]
L_kneeZ_vector = np.loadtxt(path+"L_KneeAngleZ.txt")[0:373]


L_AnkleX_vector = np.loadtxt(path+"L_AnkleAngleX.txt")[0:373]
L_AnkleY_vector = np.loadtxt(path+"L_AnkleAngleY.txt")[0:373]
L_AnkleZ_vector = np.loadtxt(path+"L_AnkleAngleZ.txt")[0:373]



R_PelvisPosX_vector_1 = np.loadtxt(path+"R1_X.txt")
R_PelvisPosY_vector_1 = np.loadtxt(path+"R1_Y.txt")
R_PelvisPosZ_vector_1 = np.loadtxt(path+"R1_Z.txt")

R_PelvisPosX_vector_2 = np.loadtxt(path+"R2_X.txt")
R_PelvisPosY_vector_2 = np.loadtxt(path+"R2_Y.txt")
R_PelvisPosZ_vector_2 = np.loadtxt(path+"R2_Z.txt")

L_PelvisPosX_vector_1 = np.loadtxt(path+"L1_X.txt")
L_PelvisPosY_vector_1 = np.loadtxt(path+"L1_Y.txt")
L_PelvisPosZ_vector_1 = np.loadtxt(path+"L1_Z.txt")

L_PelvisPosX_vector_2 = np.loadtxt(path+"L2_X.txt")
L_PelvisPosY_vector_2 = np.loadtxt(path+"L2_Y.txt")
L_PelvisPosZ_vector_2 = np.loadtxt(path+"L2_Z.txt")


a=((R_PelvisPosX_vector_1[1:]+ R_PelvisPosX_vector_2[1:]+L_PelvisPosX_vector_1[1:]+L_PelvisPosX_vector_2[1:])/4)
b=((R_PelvisPosY_vector_1[1:]+ R_PelvisPosY_vector_2[1:]+L_PelvisPosY_vector_1[1:]+L_PelvisPosY_vector_2[1:])/4)
c=((R_PelvisPosZ_vector_1[1:]+ R_PelvisPosZ_vector_2[1:]+L_PelvisPosZ_vector_1[1:]+L_PelvisPosZ_vector_2[1:])/4)


print(len(time_vector))
print(len(a))
f = open("demofile2.txt", "a")
for i in range(len(time_vector)):
	f.write("{:.5f} {:.3f} {:.3f} {:.3f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(time_vector[i],a[i],b[i],c[i],(R_PelvisX_vector[i]+L_PelvisX_vector[i])/2,(R_PelvisY_vector[i]+L_PelvisY_vector[i])/2,(R_PelvisZ_vector[i]+L_PelvisZ_vector[i])/2,R_HipX_vector[i],R_HipY_vector[i],R_HipZ_vector[i],R_kneeX_vector[i],R_kneeY_vector[i],R_kneeZ_vector[i],R_AnkleX_vector[i],R_AnkleY_vector[i],R_AnkleZ_vector[i],L_HipX_vector[i],L_HipY_vector[i],L_HipZ_vector[i],L_kneeX_vector[i],L_kneeY_vector[i],L_kneeZ_vector[i],L_AnkleX_vector[i],L_AnkleY_vector[i],L_AnkleZ_vector[i]))









# PelvisY = np.loadtxt(pathrtss2)
# PelvisY_ins = PelvisY[0:]
# PelvisY_vector.append(PelvisY_ins)


# PelvisZ = np.loadtxt(pathrtss3)
# PelvisZ_ins = PelvisZ[0:]
# PelvisZ_vector.append(PelvisZ_ins)


# pos_reward_vector = np.hstack(pos_reward_vector)
# pos_reward_mean = pos_reward_vector.mean(axis=0)
# pos_reward_std = pos_reward_vector.std(axis=0)


# # for path in pathrtss2:
# # 	cop_reward_ins = np.loadtxt(path)
# # 	# if cop_reward_ins.shape[0]>133:
# # 	cop_reward_ins = cop_reward_ins[164:-1]
# # 	cop_left_reward_vector.append(cop_reward_ins)
# # cop_left_reward_vector = np.vstack(cop_left_reward_vector)
# # cop_left_reward_mean = cop_left_reward_vector.mean(axis=0)
# # cop_left_reward_std = cop_left_reward_vector.std(axis=0)

# x_time = np.loadtxt(path1+"time_vector.txt")
# x_time = (x_time[-34:]-15.0)/1.0*100


# for path in pathrtss3:
# 	cop_reward_ins = np.loadtxt(path)
# 	# if cop_reward_ins.shape[0] > 133:
# 	cop_reward_ins = cop_reward_ins[-34:]
# 	cop_right_reward_vector.append(cop_reward_ins)
# cop_right_reward_vector = np.vstack(cop_right_reward_vector)
# cop_right_reward_mean = cop_right_reward_vector.mean(axis=0)
# cop_right_reward_std = cop_right_reward_vector.std(axis=0)




