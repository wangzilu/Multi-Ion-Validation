import json
import numpy as np
import matplotlib.pyplot as plt


# 从二进制文件读取 JSON 字符串
with open('mkm.bin', 'rb') as f:
    json_str = f.read().decode('utf-8')

# 解析 JSON 数据
results = json.loads(json_str)

calMode = 'mkm'

depth = np.arange(0.25, 350.25, 0.25)
alpha = np.zeros((len(results), depth.size), dtype=np.float32)
beta  = np.zeros((len(results), depth.size), dtype=np.float32)
Z1D   = np.zeros((len(results), depth.size), dtype=np.float32)
alpha_beta = np.zeros((len(results), depth.size, 2), dtype=np.float32)
energy = np.zeros(len(results), dtype=np.float32)

i = 0
for item in results:
    _energy = item['measEnergy']
    _depth  = item['depth']
    _alpha  = item['alpha']
    _beta   = item['beta']
    alpha[i,:np.size(_alpha)] = _alpha
    beta[i,:np.size(_beta)]   = _beta
    energy[i] = _energy / 12
    i += 1

alpha_beta[:,:,0] = alpha
alpha_beta[:,:,1] = beta

if calMode == 'mkm':
    Z1D[alpha > 0] = (alpha[alpha > 0] - 0.172) / 0.0615
    Z1D.tofile('Z1D.bin')
else:
    alpha.tofile('lem_alpha.bin')
    beta.tofile('lem_beta.bin')
    alpha_beta.tofile('lem_alpha_beta.bin')

# full_energy = np.arange(energy[0]+0.1, 430, 2.5)
# Z1D_star    = np.zeros((len(full_energy), depth.size), dtype=np.float32)

# for i in np.arange(len(full_energy)):
#     index = find_nearest_index(energy, full_energy[i])
#     for j in np.arange(len(depth)):
#         if full_energy[i] > energy[index]:
#             Z1D_star[i, j] = Z1D[index, j] + (Z1D[index + 1, j] - Z1D[index, j]) / (energy[index + 1] - energy[index]) * (full_energy[i] - energy[index])
#         else:
#             Z1D_star[i, j] = Z1D[index - 1, j] + (Z1D[index, j] - Z1D[index - 1, j]) / (energy[index] - energy[index - 1]) * (full_energy[i] - energy[index-1])
#
for i in range(0, len(results)):
    _Z1D = Z1D[i,:]
    plt.plot(depth, _Z1D)
plt.xlabel('Depth (mm)')
plt.ylabel('Z1D')
plt.title('Z1D vs Water Depth')
plt.grid(True)
plt.show()