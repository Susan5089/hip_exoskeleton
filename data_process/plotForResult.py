import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
from scipy.interpolate import interp1d
from scipy import interpolate
#pos_reward = []
#ee_reward=[]
#cop_left_reward=[]
#cop_right_reward=[]

path ="/home/shuzhen/Downloads/Hip_exoskeleton_NCSU22_old42/build/"


y_bspl=[]
x_bspl=[]
with open(path+"contact_force_vector_l_height.txt","r") as f:
	data0 = f.readlines()
data0 = [float(i.rstrip()) for i in data0]
data = data0[541:]

start_index=[]
end_index = []
for idx, x in enumerate(data[:-1]):
	if (x!=0) and data[idx+1]==0:
		start_index.append(idx)
	if (x==0) and data[idx+1]>0:
		end_index.append(idx)

valid_s_idx = []
for s_idx, e_idx in zip(start_index, end_index):
	if e_idx - s_idx <= 0:
		raise ValueError(s_idx, e_idx)
	if e_idx - s_idx > 10:
		valid_s_idx.append(s_idx)
print(valid_s_idx)
index = valid_s_idx

y_bspl0 = list()
y_bspl1 = list()
y_bspl2 = list()
y_bspl3 = list()
y_bspl4 = list()
y_bspl5 = list()
y_bspl6 = list()
y_bspl7 = list()
y_bspl8 = list()
y_bspl9 = list()
y_bspl10 = list()
y_bspl11 = list()
y_bspl12 = list()

with open(path+"contact_force_vector_r_height.txt","r") as f:
	data1 = f.readlines()
data1 = [float(i.rstrip()) for i in data1]
data1 = data1[541:]


with open(path+"hip_l_exo_angle.txt","r") as f:
	data2 = f.readlines()
data2 = [float(i.rstrip()) for i in data2]
data2 = data2[541:]


with open(path+"hip_r_exo_angle.txt","r") as f:
	data3 = f.readlines()
data3 = [float(i.rstrip()) for i in data3]
data3 = data3[541:]

with open(path+"hip_l_human_angle.txt","r") as f:
	data4 = f.readlines()
data4 = [float(i.rstrip()) for i in data4]
data4 = data4[541:]

with open(path+"hip_r_human_angle.txt","r") as f:
	data5 = f.readlines()
data5 = [float(i.rstrip()) for i in data5]
data5 = data5[541:]

with open(path+"hip_l_exo_torque.txt","r") as f:
	data6 = f.readlines()
data6 = [float(i.rstrip()) for i in data6]
data6 = data6[541:]

with open(path+"hip_r_exo_torque.txt","r") as f:
	data7 = f.readlines()
data7 = [float(i.rstrip()) for i in data7]
data7 = data7[541:]

with open(path+"hip_l_human_torque.txt","r") as f:
	data8 = f.readlines()
data8 = [float(i.rstrip()) for i in data8]
data8 = data8[541:]

   
with open(path+"hip_r_human_torque.txt","r") as f:
	data9 = f.readlines()
data9 = [float(i.rstrip()) for i in data9]
data9 = data9[541:]



for i in range(len(index)-1):
	a=np.asarray(data[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspline[y_bspline<0] = 0
	y_bspl0.append(y_bspline)
y_bspl0 = np.vstack(y_bspl0)
print(y_bspl0.shape)

mean0 = y_bspl0.mean(axis=0)
std0 = y_bspl0.std(axis=0)
std_new0 = mean0-std0
std_new0[std_new0<0] = 0


for i in range(len(index)-1):
	a=np.asarray(data1[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspline[y_bspline<0] = 0
	y_bspl1.append(y_bspline)
y_bspl1 = np.vstack(y_bspl1)
print(y_bspl1.shape)


mean1 = y_bspl1.mean(axis=0)
std1 = y_bspl1.std(axis=0)
std_new1 = mean1-std1
std_new1[std_new1<0] = 0



for i in range(len(index)-1):
	a=np.asarray(data2[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl2.append(y_bspline)
y_bspl2 = np.vstack(y_bspl2)
print(y_bspl2.shape)


mean2 = y_bspl2.mean(axis=0)
std2 = y_bspl2.std(axis=0)
std_new2 = mean2-std2



for i in range(len(index)-1):
	a=np.asarray(data3[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl3.append(y_bspline)
y_bspl3 = np.vstack(y_bspl3)
print(y_bspl3.shape)


mean3 = y_bspl3.mean(axis=0)
std3 = y_bspl3.std(axis=0)
std_new3 = mean3-std3


for i in range(len(index)-1):
	a=np.asarray(data4[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl4.append(y_bspline)
y_bspl4 = np.vstack(y_bspl4)
print(y_bspl4.shape)


mean4 = y_bspl4.mean(axis=0)
std4 = y_bspl4.std(axis=0)
std_new4 = mean4-std4



for i in range(len(index)-1):
	a=np.asarray(data5[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl5.append(y_bspline)
y_bspl5 = np.vstack(y_bspl5)
print(y_bspl5.shape)


mean5 = y_bspl5.mean(axis=0)
std5 = y_bspl5.std(axis=0)
std_new5 = mean5-std5


for i in range(len(index)-1):
	a=np.asarray(data6[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl6.append(y_bspline)
y_bspl6 = np.vstack(y_bspl6)
print(y_bspl6.shape)

mean6 = y_bspl6.mean(axis=0)
std6 = y_bspl6.std(axis=0)
std_new6 = mean6-std6


for i in range(len(index)-1):
	a=np.asarray(data7[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl7.append(y_bspline)
y_bspl7 = np.vstack(y_bspl7)
print(y_bspl7.shape)

mean7 = y_bspl7.mean(axis=0)
std7 = y_bspl7.std(axis=0)
std_new7 = mean7-std7


for i in range(len(index)-1):
	a=np.asarray(data8[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl8.append(y_bspline)
y_bspl8 = np.vstack(y_bspl8)
print(y_bspl8.shape)

mean8 = y_bspl8.mean(axis=0)
std8 = y_bspl8.std(axis=0)
std_new8 = mean8-std8

for i in range(len(index)-1):
	a=np.asarray(data9[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl9.append(y_bspline)
y_bspl9 = np.vstack(y_bspl9)
print(y_bspl9.shape)

mean9 = y_bspl9.mean(axis=0)
std9 = y_bspl9.std(axis=0)
std_new9 = mean9-std9



plt.figure(0)
plt.plot(y_new, mean0, color="blue", linewidth=1.5,  label="Right foot GRF")
plt.plot(y_new, mean1, color="magenta", linewidth=1.5,  label="Left foot GRF")
plt.xlabel("Gait cycle/(%)", fontsize=13)
plt.ylabel("Foot GRF/(N)", fontsize=13)
plt.fill_between(y_new, mean0+std0, std_new0, facecolor='blue', alpha=0.2)
plt.fill_between(y_new, mean1+std1, std_new1, facecolor='magenta', alpha=0.2)
plt.legend(loc='best',fontsize=11)
plt.show()


#
plt.figure(1)
plt.xlabel("Gait cycle/(%)", fontsize=13)
plt.ylabel("Joint angle/(deg)", fontsize=13)
plt.plot(y_new, mean2, color="black", linewidth=2.5,  linestyle='dotted', label="hip_l_exo")
plt.plot(y_new, mean3, color="blue", linewidth=1.5,  linestyle="--", label="hip_r_exo")
# plt.plot(y_new, mean4, color="red", linewidth=1.5,  linestyle="-.", label="hip_l_human")
# plt.plot(y_new, mean5, color="magenta", linewidth=1.5, label="hip_r_human")

# plt.fill_between(y_new, mean2+std2, std_new2, facecolor='black', alpha=0.2)
# plt.fill_between(y_new, mean3+std3, std_new3, facecolor='blue', alpha=0.2)
# plt.fill_between(y_new, mean4+std4, std_new4, facecolor='red', alpha=0.2)
# plt.fill_between(y_new, mean5+std5, std_new5, facecolor='magenta', alpha=0.2)
plt.legend(loc='best',fontsize=11)
plt.show()


plt.figure(2)
plt.xlabel("Gait cycle/(%)", fontsize=13)
plt.ylabel("Joint torque/(N*m)", fontsize=13)
plt.plot(y_new, mean6, color="black", linewidth=2.5,  linestyle='dotted', label="hip_l_exo ")
plt.plot(y_new, mean7, color="blue", linewidth=1.5,  linestyle="--", label="hip_r_exo")

# plt.fill_between(y_new, mean6, std_new6, facecolor='black', alpha=0.2)
# plt.fill_between(y_new, mean7, std_new7, facecolor='blue', alpha=0.2)


plt.figure(3)
plt.xlabel("Gait cycle/(%)", fontsize=13)
plt.ylabel("Joint torque/(N*m)", fontsize=13)
plt.plot(y_new, -mean8, color="black", linewidth=2.5,  linestyle='dotted', label="hip_l_human ")
plt.plot(y_new, -mean9, color="blue", linewidth=1.5,  linestyle="--", label="hip_r_human")

plt.legend(loc='best',fontsize=11)
plt.show()












