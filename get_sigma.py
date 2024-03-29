import os
import torch
import pdb
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import time

file_dir_v1 = '/Users/yufanwang/Desktop/Study/Project/Data/backup/save_tensor/'
file_list = sorted(os.listdir(file_dir_v1))
idx_list_v1 = []
for file in file_list:
    if "save_tensor" in file or ".DS_St" in file:
        continue
    idx_list_v1.append(int(file[:-3]))
# print(sorted(idx_list_v1))

file_dir_v2 = '/Users/yufanwang/Desktop/Study/Project/Data/backup/save_tensor_v2/'
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
mean_rate_v1 = 0
mean_rate_v2 = 0
cnt = 0
final_idx = sorted(final_idx)
for id in tqdm(final_idx[1000:]):
    cnt += 1
    data_v1 = torch.load(file_dir_v1+str(id)+".pt",map_location=torch.device('cpu'))-1
    data_v2 = torch.load(file_dir_v2+str(id)+".pt",map_location=torch.device('cpu'))-1
    # data_v1 = torch.load(file_dir_v1+str(id)+".pt")-1
    # data_v2 = torch.load(file_dir_v2+str(id)+".pt")-1
    
    test_v1 = data_v1
    test_v1[test_v1 < 0] = 0
    test_v1[test_v1 > 0] = 1
    print(torch.norm(test_v1,p=1)/196608)
    mean_rate_v1 += torch.norm(test_v1,p=1)/196608
    test_v2 = data_v2
    test_v2[test_v2 < 0] = 0
    test_v2[test_v2 > 0] = 1
    print(torch.norm(test_v2,p=1)/196608)
    mean_rate_v2 += torch.norm(test_v2,p=1)/196608
    # pdb.set_trace()
    matrix = data_v1 / data_v2*0.1
    
    value = float(torch.mean(matrix).float())
    if value == float('nan'):
        pdb.set_trace()
    elif value == float('inf') or value == float('-inf'):
        # pdb.set_trace()
        continue
    elif value > 1000 or value < -1000:
        continue
    else:
        mean_sigma.append(value)

print(mean_rate_v1/cnt)
print(mean_rate_v2/cnt)
# time.sleep(10)
pdb.set_trace()

mean_sigma_clean=[elem if not np.isnan(elem) else None for elem in mean_sigma]
while None in mean_sigma_clean:
	mean_sigma_clean.remove(None)
# pdb.set_trace()
data = np.array(mean_sigma_clean)
print(len(data))
fig, ax = plt.subplots()
# pdb.set_trace()
hist = ax.hist(data, bins=3000, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='none')

for i in range(len(hist[0])):
        if hist[0][i] == max(hist[0]):
            plt.annotate(text=hist[1][i], xy=(hist[1][i] + 0.1 / 3, hist[0][i]))
plt.xlim([-10,20])
plt.xlabel("$\sigma'$")
plt.ylabel("Frequency")
plt.savefig("sigma.png")
plt.show()




