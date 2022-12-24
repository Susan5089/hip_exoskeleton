import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from scipy.interpolate import interp1d
from scipy import interpolate
#pos_reward = []
#ee_reward=[]
#cop_left_reward=[]
#cop_right_reward=[]


path =  "/home/shuzhen/Exo_human_walk_test4_human6_copy2/build/"
path1 = "/home/shuzhen/Exo_human_walk_test4_human6_copy2/build/result2/"

index=[]
y_bspl=[]
x_bspl=[]



with open(path+"contact_force_vector_r_height.txt","r") as f:
	data0 = f.readlines()
data0 = [float(i.rstrip()) for i in data0]
data = data0[234:]

for idx, x in enumerate(data[:-1]):
	if (x==0) and data[idx+1]>0:
		index.append(idx)
print(index)
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
y_bspl13 = list()
y_bspl14 = list()
y_bspl15 = list()

with open(path+"contact_force_vector_l_height.txt","r") as f:
	data1 = f.readlines()
data1 = [float(i.rstrip()) for i in data1]
data1 = data1[234:]


with open(path1+"L_Adductor_Magnus.txt","r") as f:
	data2 = f.readlines()
data2 = [float(i.rstrip()) for i in data2]
data2 = data2[234:]


with open(path1+"L_Flexor_Hallucis.txt","r") as f:
	data3 = f.readlines()
data3 = [float(i.rstrip()) for i in data3]
data3 = data3[234:]

with open(path1+"L_Gastrocnemius_Medial_Head.txt","r") as f:
	data4 = f.readlines()
data4 = [float(i.rstrip()) for i in data4]
data4 = data4[234:]

with open(path1+"L_Gluteus_Maximus.txt","r") as f:
	data5 = f.readlines()
data5 = [float(i.rstrip()) for i in data5]
data5 = data5[234:]

with open(path1+"L_Gluteus_Medius.txt","r") as f:
	data6 = f.readlines()
data6 = [float(i.rstrip()) for i in data6]
data6 = data6[234:]

with open(path1+"L_iliacus.txt","r") as f:
	data7 = f.readlines()
data7 = [float(i.rstrip()) for i in data7]
data7 = data7[234:]

with open(path1+"L_Psoas_Major.txt","r") as f:
	data8 = f.readlines()
data8 = [float(i.rstrip()) for i in data8]
data8 = data8[234:]


with open(path1+"L_Rectus_Femoris.txt","r") as f:
	data9 = f.readlines()
data9 = [float(i.rstrip()) for i in data9]
data9 = data9[234:]

with open(path1+"L_Semimembranosus.txt","r") as f:
	data10 = f.readlines()
data10 = [float(i.rstrip()) for i in data10]
data10 = data10[234:]


with open(path1+"L_Soleus.txt","r") as f:
	data11 = f.readlines()
data11 = [float(i.rstrip()) for i in data11]
data11 = data11[234:]


with open(path1+"L_Tibialis_Posterior.txt","r") as f:
	data12 = f.readlines()
data12 = [float(i.rstrip()) for i in data12]
data12 = data12[234:]


with open(path1+"L_Vastus_Intermedius.txt","r") as f:
	data13 = f.readlines()
data13 = [float(i.rstrip()) for i in data13]
data13 = data13[234:]


with open(path1+"L_Vastus_Lateralis.txt","r") as f:
	data14 = f.readlines()
data14 = [float(i.rstrip()) for i in data14]
data14 = data14[234:]


with open(path1+"L_Vastus_Medialis.txt","r") as f:
	data15 = f.readlines()
data15 = [float(i.rstrip()) for i in data15]
data15 = data15[234:]
data_Vastus_Inter = np.array(data13)
data_Vastus_Lateral = np.array(data14)
data_Vastus_Media= np.array(data15)

data15 =list((data_Vastus_Inter+data_Vastus_Lateral +data_Vastus_Media)/3)



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

#
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


for i in range(len(index)-1):
	a=np.asarray(data10[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl10.append(y_bspline)
y_bspl10 = np.vstack(y_bspl10)
print(y_bspl10.shape)

mean10 = y_bspl10.mean(axis=0)
std10 = y_bspl10.std(axis=0)
std_new10 = mean10-std10


for i in range(len(index)-1):
	a=np.asarray(data11[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl11.append(y_bspline)
y_bspl11 = np.vstack(y_bspl11)
print(y_bspl11.shape)

mean11 = y_bspl11.mean(axis=0)
std11 = y_bspl11.std(axis=0)
std_new11 = mean11-std11

for i in range(len(index)-1):
	a=np.asarray(data12[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl12.append(y_bspline)
y_bspl12 = np.vstack(y_bspl12)
print(y_bspl12.shape)

mean12 = y_bspl12.mean(axis=0)
std12 = y_bspl12.std(axis=0)
std_new12 = mean12-std12

# for i in range(len(index)-1):
# 	a=np.asarray(data13[index[i]:index[i+1]])
# 	y=np.linspace(0,100,a.shape[0])
# 	y_new = np.linspace(0,100,100)
# 	tck = interpolate.splrep(y, a)
# 	y_bspline = interpolate.splev(y_new, tck)
# 	y_bspl13.append(y_bspline)
# y_bspl13 = np.vstack(y_bspl13)
# print(y_bspl13.shape)
#
# mean13 = y_bspl13.mean(axis=0)
# std13 = y_bspl13.std(axis=0)
# std_new13 = mean13-std13
#
#
# for i in range(len(index)-1):
# 	a=np.asarray(data14[index[i]:index[i+1]])
# 	y=np.linspace(0,100,a.shape[0])
# 	y_new = np.linspace(0,100,100)
# 	tck = interpolate.splrep(y, a)
# 	y_bspline = interpolate.splev(y_new, tck)
# 	y_bspl14.append(y_bspline)
# y_bspl14 = np.vstack(y_bspl14)
# print(y_bspl14.shape)
#
# mean14 = y_bspl14.mean(axis=0)
# std14 = y_bspl14.std(axis=0)
# std_new14 = mean14-std14


for i in range(len(index)-1):
	a=np.asarray(data15[index[i]:index[i+1]])
	y=np.linspace(0,100,a.shape[0])
	y_new = np.linspace(0,100,100)
	tck = interpolate.splrep(y, a)
	y_bspline = interpolate.splev(y_new, tck)
	y_bspl15.append(y_bspline)
y_bspl15 = np.vstack(y_bspl15)
print(y_bspl15.shape)

mean15 = y_bspl15.mean(axis=0)
std15= y_bspl15.std(axis=0)
std_new15 = mean15-std15


fig, (ax1, ax2, ax3, ax4,ax5,ax6, ax7, ax8, ax9,ax10) = plt.subplots(10)

fig.text(0.5, 0.04, 'Gait cycle/(%)', ha='center',fontsize=12)
fig.text(0.04, 0.5, 'Muscle activation', va='center', rotation='vertical',fontsize=12)
ax1.plot(y_new, mean2, color="black", linewidth=1.5,  linestyle='dotted', label="Adductor_Magnus")
ax1.legend(loc='upper right',fontsize=11)
# ax2.plot(y_new, mean3, color="blue", linewidth=1.5,  linestyle="--", label="femur force")
ax2.plot(y_new, mean4, color="red", linewidth=1.5,   label="Gastrocnemius_Medial_Head")
ax2.legend(loc='upper right',fontsize=11)
ax3.plot(y_new, mean5, color="cyan", linewidth=1.5,  linestyle='-.', label="Gluteus_Maximus")
ax3.legend(loc='upper right',fontsize=11)
ax4.plot(y_new, mean6, color="green", linewidth=1.5,  linestyle="--", label="Gluteus_Medius")
ax4.legend(loc='upper right',fontsize=11)
ax5.plot(y_new, mean7, color="black", linewidth=1.5,  linestyle="dotted", label="Iliacus")
ax5.legend(loc='upper right',fontsize=11)
ax6.plot(y_new, mean8, color="magenta", linewidth=1.5, label="Psoas_Major")
ax6.legend(loc='upper right',fontsize=11)
ax7.plot(y_new, mean9, color="black", linewidth=1.5,  linestyle="--", label="Rectus_Femoris")
ax7.legend(loc='upper right',fontsize=11)
ax8.plot(y_new, mean10, color="red", linewidth=1.5,  linestyle="-.", label="Semimembranosus")
ax8.legend(loc='upper right',fontsize=11)
ax9.plot(y_new, mean11, color="cyan", linewidth=1.5,  linestyle="dotted", label="Soleus")
ax9.legend(loc='upper right',fontsize=11)
# ax11.plot(y_new, mean12, color="green", linewidth=1.5,  linestyle="--", label="Tibialis_Posterior")
# ax12.plot(y_new, mean13, color="black", linewidth=1.5,  linestyle="--", label="Vastus_Intermedius")
# ax10.plot(y_new, mean14, color="magenta", linewidth=1.5,  linestyle="-.", label="Vastus_Lateralis")
# ax10.legend(loc='upper right',fontsize=11)
ax10.plot(y_new, mean15, color="red", linewidth=1.5,  label="Vastus")
ax10.legend(loc='upper right',fontsize=11)

ax1.fill_between(y_new, mean2+std2, std_new2, facecolor='black', alpha=0.2)
ax2.fill_between(y_new, mean4+std4, std_new4, facecolor='red', alpha=0.2)
ax3.fill_between(y_new, mean5+std5, std_new5, facecolor='cyan', alpha=0.2)
ax4.fill_between(y_new, mean6+std6, std_new6, facecolor='green', alpha=0.2)
ax5.fill_between(y_new, mean7+std7, std_new7, facecolor='black', alpha=0.2)
ax6.fill_between(y_new, mean8+std8, std_new8, facecolor='magenta', alpha=0.2)
ax7.fill_between(y_new, mean9+std9, std_new9, facecolor='black', alpha=0.2)
ax8.fill_between(y_new, mean10+std10, std_new10, facecolor='red', alpha=0.2)
ax9.fill_between(y_new, mean11+std11, std_new11, facecolor='cyan', alpha=0.2)
# ax10.fill_between(y_new, mean14+std14, std_new14, facecolor='magenta', alpha=0.2)
ax10.fill_between(y_new, mean15+std15, std_new15, facecolor='red', alpha=0.2)

plt.show()


#
