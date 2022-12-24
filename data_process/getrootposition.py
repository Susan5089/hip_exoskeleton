import matplotlib.pyplot as plt
import numpy as np 
from glob import glob
from scipy.interpolate import interp1d

#pos_reward = []
#ee_reward=[]
#cop_left_reward=[]
#cop_right_reward=[]


path = "/home/shuzhen/Hip_exoskeleton_NCSU/WBDS01walkO01Cmkr.txt"



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




