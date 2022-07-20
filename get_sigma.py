import os
import torch
import pdb
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np

file_dir_v1 = '/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v008/save_tensor/'
file_list = sorted(os.listdir(file_dir_v1))
idx_list_v1 = []
for file in file_list:
    if "save_tensor" in file or ".DS_St" in file:
        continue
    idx_list_v1.append(int(file[:-3]))
# print(sorted(idx_list_v1))

file_dir_v2 = '/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v008/save_tensor_v2/'
file_list = sorted(os.listdir(file_dir_v2))
idx_list_v2 = []
for file in file_list:
    if "save_tensor" in file or ".DS_St" in file:
        continue
    idx_list_v2.append(int(file[:-3]))
# print(sorted(idx_list_v2))

final_idx = []
for i in idx_list_v2:
    if i in idx_list_v1:
        final_idx.append(i)

mean_sigma = []
for id in tqdm(final_idx[1000:]):
    # data_v1 = torch.load(file_dir_v1+str(id)+".pt",map_location=torch.device('cpu'))-1
    # data_v2 = torch.load(file_dir_v2+str(id)+".pt",map_location=torch.device('cpu'))-1
    data_v1 = torch.load(file_dir_v1+str(id)+".pt")-1
    data_v2 = torch.load(file_dir_v2+str(id)+".pt")-1
    matrix = data_v1 / data_v2*0.1
    value = float(torch.mean(matrix).float())
    if value == float('nan'):
        pdb.set_trace()
    elif value == float('inf'):
        # pdb.set_trace()
        continue
    else:
        mean_sigma.append(value)


# pdb.set_trace()
data = np.array(mean_sigma)
# # plt.hist(data)
# plt.hist(data, bins=3000, alpha=0.5,
#          histtype='stepfilled', color='steelblue',
#          edgecolor='none')
# plt.xlim([-10,20])
# plt.show()

fig, ax = plt.subplots()
pdb.set_trace()
hist = ax.hist(data, bins=3000, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')
# pdb.set_trace()

for i in range(len(hist[0])):
        #  xy 即为在图上的坐标  text 为内容
        if hist[0][i] == max(hist[0]):
            plt.annotate(text=hist[1][i], xy=(hist[1][i] + 0.1 / 3, hist[0][i]))
plt.xlim([-10,20])
plt.savefig("sigma.png")
plt.show()




