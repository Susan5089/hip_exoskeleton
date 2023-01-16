inputFile = "/home/shuzhen/Downloads/walk_mannual.mot"
outputFile = "/home/shuzhen/Exo_human_walk_test/data/motion/NJIT_BME_Leg_Exo_Step_Walk.mot"
outputFile2 = "/home/shuzhen/Exo_human_walk_test/data/motion/NJIT_BME_Leg_Exo_Step_Walk2.mot"

# with open(inputFile, 'r') as f:
# 	data = f.readlines()

# discardIndex = list(range(1,7)) + list(range(13,22)) + list(range(24,32)) + list(range(34,42)) + list(range(43,45)) + list(range(46,54)) + list(range(55,57))+ list(range(58,79))
# print(discardIndex)
# with open(outputFile, 'w') as f: 
# 	for i in range(7):
# 		f.write(data[i])
# 	data = data[7:]
# 	data = [i.rstrip() for i in data]
# 	for i in range(len(data)):
# 		if i%80 in discardIndex:
# 			continue
# 		f.write(data[i] + " ")
# 		if (i+1)%80==0:
# 			f.write("\n")

with open(outputFile, 'r') as f:
	data = f.readlines()
data = data[1:]
import numpy as np 
a =np.zeros((488, 34))
for idx, line in enumerate(data):
	a[idx] = np.asarray([float(i) for i in line.rstrip().split()])
print(a)

output = a[:,:-16]
# output =[]
# for i in range(a.shape[0]-1):
# 	output.append(np.linspace(a[i], a[i+1], num=49, endpoint=False))
# output.append(np.expand_dims(a[-1], axis=0))




output = np.vstack(output)
print(output.shape)
with open(outputFile2, 'w') as f:
	for line in output:
		for idx, num in enumerate(line):
			f.write("{:05f}".format(num)) 
			if idx != len(line)-1:
				f.write("\t")
		f.write("\n")

		# f.write(" ".join([str(i) for i in list(line)]) + "\n") 
# tmp = np.linspace(a[0], a[1], num=32, endpoint=True)
# print(tmp.shape)
# print(tmp[0] == a[0])
# print(tmp[-1] == a[1])