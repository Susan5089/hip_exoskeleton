import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
from scipy.interpolate import interp1d

#pos_reward = []
#ee_reward=[]
#cop_left_reward=[]
#cop_right_reward=[]

# treadmill 1m/s 
# path_ang = "/home/shuzhen/5722711/WBDS01walkT04ang.txt"   #treadmill 
path_ang = "/home/shuzhen/5722711/WBDS01walkOSang.txt"  #overground 

path_mkr = "/home/shuzhen/5722711/WBDS01walkT01mkr.txt"
path_out = "/home/shuzhen/Hip_exoskeleton_NCSU/data/motion/human_gt_treadmill_1ms.txt"
with open(path_ang, "r") as f:
    ang_cont = f.readlines()
    ang_cont = ang_cont[1:]

with open(path_ang+".tmp", "w") as f:
    for i in ang_cont:
        f.write(i)
ang_np = np.loadtxt(path_ang+".tmp")
x_offset = 1
y_offset = 2
z_offset = 3
x_a = (ang_np[:, x_offset] + ang_np[:, x_offset+3])/2.0*0.0 
y_a = (ang_np[:, y_offset] + ang_np[:, y_offset+3])/2.0*0.0 
z_a = (ang_np[:, z_offset] + ang_np[:, z_offset+3])/2.0*0.0 


ang_info = []
for line in ang_cont: 
    ang_info.append(line.split()[:-6])

with open(path_mkr, "r") as f:
    mkr_cont = f.readlines()
    mkr_cont = mkr_cont[1:]
with open(path_mkr+".tmp", "w") as f:
    for i in mkr_cont:
        f.write(i)

mkr_np = np.loadtxt(path_mkr+".tmp")
x_offset = 1
y_offset = 2
z_offset = 3
x = (mkr_np[:, x_offset] + mkr_np[:, x_offset+3] + mkr_np[:, x_offset+6] + mkr_np[:, x_offset+9])/4.0 *0.0 
y = (mkr_np[:, y_offset] + mkr_np[:, y_offset+3] + mkr_np[:, y_offset+6] + mkr_np[:, y_offset+9])/4.0 *0.0
z = (mkr_np[:, z_offset] + mkr_np[:, z_offset+3] + mkr_np[:, z_offset+6] + mkr_np[:, z_offset+9])/4.0 *0.0

out_np = np.zeros([101, 25])
out_np[:, 0] = ang_np[:, 0][:101]
out_np[:, 1] = x[:101]
out_np[:, 2] = y[:101]
out_np[:, 3] = z[:101]
out_np[:, 4] = x_a[:101]
out_np[:, 5] = y_a[:101]
out_np[:, 6] = z_a[:101]
out_np[:, 7:] = ang_np[:, 7:-6]

# FemurL_X FemurL_Y FemurL_Z 
out_np[:, 10:13] = np.concatenate((out_np[50:100, 10:13], out_np[0:51, 10:13]), axis=0)
# TibiaL_X TibiaL_Y TibiaL_Z
out_np[:, 16:19] = np.concatenate((out_np[50:100, 16:19], out_np[0:51, 16:19]), axis=0)
# TalusL_X TalusL_Y TalusL_Z
out_np[:, 22:] = np.concatenate((out_np[50:100, 22:], out_np[0:51, 22:]), axis=0)
np.savetxt(path_out, out_np, fmt='%.5f') 


overhead = "Units are S.I. units (second, meters, Newtons, ...)\n\
Angles are in degrees.\n\
name walk\n\
datarows 101 \n\
datacolumns 25 \n\
range 0  2.0\n\
endheader\n\
time Pelvis_PX Pelvis_PY Pelvis_PZ Pelvis_X Pelvis_Y Pelvis_Z FemurR_X FemurR_Y FemurR_Z FemurL_X FemurL_Y FemurL_Z TibiaR_X TibiaR_Y TibiaR_Z TibiaL_X TibiaL_Y TibiaL_Z TalusR_X TalusR_Y \
TalusR_Z TalusL_X TalusL_Y TalusL_Z\n"

with open(path_out, "r") as f:
    cont = f.read() 
with open(path_out, "w") as f:
    f.write(overhead)
    f.write(cont[:-1]) 


# with open(path_out, "w") as f:
#     for i in range(len(ang_info)):
#         if i != len(ang_info)-1:
#             f.write("{:.4f} {:.4f} {:.4f} {:.4f} {}\n".format(float(ang_info[i][0]), x[i], y[i], z[i], " ".join(ang_info[i][1:])))
#         else:
#             f.write("{:.4f} {:.4f} {:.4f} {:.4f} {}".format(float(ang_info[i][0]), x[i], y[i], z[i], " ".join(ang_info[i][1:])))
exit()




f = open("/home/shuzhen/Hip_exoskeleton_NCSU/WBDS01walkO01Cmkr.txt", "r")

list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
list7=[]
list8=[]
list9=[]
list10=[]
list11=[]
list12=[]


lines =f.readlines()
for line in lines:
    a =line.split()
    x=a[1]
    list1.append(x)



for line in lines:
    a =line.split()
    x=a[2]
    list2.append(x)


for line in lines:
    a =line.split()
    x=a[3]
    list3.append(x)


for line in lines:
    a =line.split()
    x=a[4]
    list4.append(x)



for line in lines:
    a =line.split()
    x=a[5]
    list5.append(x)


for line in lines:
    a =line.split()
    x=a[6]
    list6.append(x)



for line in lines:
    a =line.split()
    x=a[7]
    list7.append(x)


for line in lines:
    a =line.split()
    x=a[8]
    list8.append(x)


for line in lines:
    a =line.split()
    x=a[9]
    list9.append(x)



for line in lines:
    a =line.split()
    x=a[10]
    list10.append(x)


for line in lines:
    a =line.split()
    x=a[11]
    list11.append(x)


for line in lines:
    a =line.split()
    x=a[12]
    list12.append(x)

f.close()




f1 = open("R1_X.txt", "a")
for i in range(len(list1)-1):
	f1.write("{}\n".format(list1[i+1]))



f2 = open("R1_Y.txt", "a")
for i in range(len(list2)-1):
	f2.write("{}\n".format(list2[i+1]))



f3 = open("R1_Z.txt", "a")
for i in range(len(list3)-1):
	f3.write("{}\n".format(list3[i+1]))


f4 = open("L1_X.txt", "a")
for i in range(len(list4)-1):
	f4.write("{}\n".format(list4[i+1]))

f5 = open("L1_Y.txt", "a")
for i in range(len(list5)-1):
	f5.write("{}\n".format(list5[i+1]))

f6 = open("L1_Z.txt", "a")
for i in range(len(list6)-1):
	f6.write("{}\n".format(list6[i+1]))



f7 = open("R2_X.txt", "a")
for i in range(len(list7)-1):
	f7.write("{}\n".format(list7[i+1]))


f8 = open("R2_Y.txt", "a")
for i in range(len(list8)-1):
	f8.write("{}\n".format(list8[i+1]))


f9 = open("R2_Z.txt", "a")
for i in range(len(list9)-1):
	f9.write("{}\n".format(list9[i+1]))


f10 = open("L2_X.txt", "a")
for i in range(len(list10)-1):
	f10.write("{}\n".format(list10[i+1]))

f11 = open("L2_Y.txt", "a")
for i in range(len(list11)-1):
	f11.write("{}\n".format(list11[i+1]))

f12 = open("L2_Z.txt", "a")
for i in range(len(list12)-1):
	f12.write("{}\n".format(list12[i+1]))




