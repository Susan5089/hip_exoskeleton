import matplotlib.pyplot as plt
import numpy as np 

# import os


with open("r_foot_ground_ShapeNode_0.txt","r") as f:
	data0 = f.readlines()
data0 = list(set(data0))
data0.sort(key=lambda x: float(x.split()[0]))
# print(data0)

with open("r_foot_ground_ShapeNode_1.txt","r") as f:
	data1 = f.readlines()
data1 = list(set(data1))
data1.sort(key=lambda x: float(x.split()[0]))
# print(data1)

with open("r_foot_ground_ShapeNode_2.txt","r") as f:
	data2= f.readlines()
data2 = list(set(data2))
data2.sort(key=lambda x: float(x.split()[0]))

with open("r_foot_ground_ShapeNode_3.txt","r") as f:
	data3 = f.readlines()
data3 = list(set(data3))
data3.sort(key=lambda x: float(x.split()[0]))
# print(data3)

with open("l_foot_ground_ShapeNode_0.txt","r") as f:
	data4 = f.readlines()
data4 = list(set(data4))
data4.sort(key=lambda x: float(x.split()[0]))

with open("l_foot_ground_ShapeNode_1.txt","r") as f:
	data5 = f.readlines()
data5 = list(set(data5))
data5.sort(key=lambda x: float(x.split()[0]))

with open("l_foot_ground_ShapeNode_2.txt","r") as f:
	data6 = f.readlines()
data6 = list(set(data6))
data6.sort(key=lambda x: float(x.split()[0]))

with open("l_foot_ground_ShapeNode_3.txt","r") as f:
	data7 = f.readlines()
data7 = list(set(data7))
data7.sort(key=lambda x: float(x.split()[0]))


time0,x0,y0,z0 = [],[],[],[]
for line in data0:
	a = line.rstrip().split()
	time0.append(float(a[0]))
	x0.append(float(a[1])*100)
	y0.append((float(a[2])+0.86927)*100)
	z0.append(float(a[3])*100)

time1,x1,y1,z1 = [],[],[],[]
for line in data1:
	a = line.rstrip().split()
	time1.append(float(a[0]))
	x1.append(float(a[1])*100)
	y1.append((float(a[2])+0.86927)*100)
	z1.append(float(a[3])*100)

time2,x2,y2,z2 = [],[],[],[]
for line in data2:
	a = line.rstrip().split()
	time2.append(float(a[0]))
	x2.append(float(a[1])*100)
	y2.append((float(a[2])+0.86927)*100)
	z2.append(float(a[3])*100)

time3,x3,y3,z3 = [],[],[],[]
for line in data3:
	a = line.rstrip().split()
	time3.append(float(a[0]))
	x3.append(float(a[1])*100)
	y3.append((float(a[2])+0.86927)*100)
	z3.append(float(a[3])*100)



time4,x4,y4,z4 = [],[],[],[]
for line in data4:
	a = line.rstrip().split()
	time4.append(float(a[0]))
	x4.append(float(a[1])*100)
	y4.append((float(a[2])+0.86927)*100)
	z4.append(float(a[3])*100)

time5,x5,y5,z5 = [],[],[],[]
for line in data5:
	a = line.rstrip().split()
	time5.append(float(a[0]))
	x5.append(float(a[1])*100)
	y5.append((float(a[2])+0.86927)*100)
	z5.append(float(a[3])*100)

time6,x6,y6,z6 = [],[],[],[]
for line in data6:
	a = line.rstrip().split()
	time6.append(float(a[0]))
	x6.append(float(a[1])*100)
	y6.append((float(a[2])+0.86927)*100)
	z6.append(float(a[3])*100)

time7,x7,y7,z7 = [],[],[],[]
for line in data7:
	a = line.rstrip().split()
	time7.append(float(a[0]))
	x7.append(float(a[1])*100)
	y7.append((float(a[2])+0.86927)*100)
	z7.append(float(a[3])*100)






plt.figure(0)
plt.subplot(4,1,1)
plt.ylabel("Y/cm", fontsize=13)
# plt.xlabel("X/m", fontsize=13)
plt.title("right foot position")
plt.plot(time0,x0, color="blue", linewidth=1,  linestyle="-", label="r_foot_end1")
plt.plot(time0,x1, color="green", linewidth=1,  linestyle="-", label="r_foot_front1")
plt.plot(time0,x2, color="black", linewidth=1,  linestyle="-", label="r_foot_end2")
plt.plot(time0,x3, color="red", linewidth=1,  linestyle="-", label="r_foot_front2")


plt.subplot(4,1,2)
plt.ylabel("Y/cm", fontsize=13)
plt.plot(time0,y0, color="blue", linewidth=1,  linestyle="-", label="r_foot_end1")
plt.plot(time0,y1, color="green", linewidth=1,  linestyle="-", label="r_foot_front1")
plt.plot(time0,y2, color="black", linewidth=1,  linestyle="-", label="r_foot_end2")
plt.plot(time0,y3, color="red", linewidth=1,  linestyle="-", label="r_foot_front2")


# plt.plot(x1,y1, color="blue", linewidth=1,  linestyle="-", label="r_foot_front1")
# plt.plot(x3,y3, color="green", linewidth=1,  linestyle="-", label="r_foot_front2")
# plt.plot(x0,y0, color="black", linewidth=1,  linestyle="-", label="r_foot_end1")
# plt.plot(x2,y2, color="red", linewidth=1,  linestyle="-", label="r_foot_end2")
plt.legend(loc='upper left',fontsize=12)

plt.subplot(4,1,3)
plt.ylabel("Y/cm", fontsize=13)
plt.title("left foot position")
plt.plot(time0,x4, color="blue", linewidth=1,  linestyle="-", label="l_foot_end1")
plt.plot(time0,x5, color="green", linewidth=1,  linestyle="-", label="l_foot_front1")
plt.plot(time0,x6, color="black", linewidth=1,  linestyle="-", label="l_foot_end2")
plt.plot(time0,x7, color="red", linewidth=1,  linestyle="-", label="l_foot_front2")

plt.subplot(4,1,4)
plt.xlabel("Time/s", fontsize=13)
plt.ylabel("Y/cm", fontsize=13)
plt.plot(time0,y4, color="blue", linewidth=1,  linestyle="-", label="l_foot_end1")
plt.plot(time0,y5, color="green", linewidth=1,  linestyle="-", label="l_foot_front1")
plt.plot(time0,y6, color="black", linewidth=1,  linestyle="-", label="l_foot_end2")
plt.plot(time0,y7, color="red", linewidth=1,  linestyle="-", label="l_foot_front2")
# plt.plot(x5,y5, color="blue", linewidth=1,  linestyle="-", label="l_foot_front1")
# plt.plot(x7,y7, color="green", linewidth=1,  linestyle="-", label="l_foot_front2")
# plt.plot(x4,y4, color="black", linewidth=1,  linestyle="-", label="l_foot_end1")
# plt.plot(x6,y6, color="red", linewidth=1,  linestyle="-", label="l_foot_end2")
plt.legend(loc='upper left',fontsize=12)


plt.show()
