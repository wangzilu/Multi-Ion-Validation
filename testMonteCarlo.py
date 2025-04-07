import os
from turtledemo.penrose import start
from matplotlib import pyplot as plt
import numpy as np
import time
print(os.getcwd())
from protonMC_CUDA import cudaProtonMonteCarlo
from carbonMC import carbonMonteCarlo

beamParam         = np.load('./input/beamsparam.npy')
beamParam[0][0]   = 10000000
beamParam[0][4:9] = 0
beamParam[0][3]   = 4800

sourcePos = np.load('./input/sourcepos.npy')
bmdir     = np.load('./input/bmdir.npy')
res       = 3.0

corner        = np.load('./input/corner.npy')
resolution    = np.load('./input/resolution.npy')
resolution[0] = res
resolution[1] = res
resolution[2] = res

dims    = np.load('./input/dims.npy')
dims[0] = int(dims[0] * 3 / res)
dims[1] = int(dims[1] * 3 / res)
dims[2] = int(dims[2] * 3 / res)

ctdata      = np.zeros(dims[0] * dims[1] * dims[2])
calROIIndex = np.arange(dims[0] * dims[1] * dims[2])

isocenter           = np.load('./input/isocenter.npy')
isocenter[1]        = corner[1] - resolution[1] / 2
materialComposition = np.load('./input/material.npy')
materialComposition = materialComposition[:25, :]
HUDensity           = np.array([[0., 1.]])
eneProb             = np.load('./input/eneProb.npy')

finalDose   = np.zeros(dims, dtype=np.float32, order="F")
finalDose1  = np.zeros((dims[2], dims[1], dims[0], 8), dtype=np.float32)
tempSumDose = np.zeros((dims[0] * dims[1] * dims[2] * 3), dtype=np.float32)
tempLET     = np.zeros((dims[0] * dims[1] * dims[2] * 4), dtype=np.float32)

spotDose    = np.zeros(dims, dtype=np.float32, order="F")
spotInd     = np.zeros(dims, dtype=np.uint32, order="F")

vSAD        = 20.0
start = time.time()
a = cudaProtonMonteCarlo(finalDose, finalDose1, tempSumDose, tempLET, './mc_config', beamParam, sourcePos, bmdir, ctdata, corner, resolution, dims, isocenter, materialComposition, HUDensity, eneProb, vSAD, 1, 1)
end = time.time()
print(end - start)

for i in range(8):
    tmpFinalDose = finalDose1[:, :, :, i]
    tmpIDD = np.sum(tmpFinalDose, axis=2)
    tmpIDD = np.sum(tmpIDD, axis=0)
    x = np.arange(len(tmpIDD))
    plt.plot(x, tmpIDD)

# maxUncertainty = carbonMonteCarlo(finalDose, spotDose, spotInd,
#                                   './config.txt', calROIIndex,
#                                   beamParam, sourcePos, vSAD, bmdir,
#                                   ctdata,
#                                   corner - 0.5 * resolution,
#                                   resolution, dims,
#                                   np.array(isocenter),
#                                   materialComposition,
#                                   HUDensity, eneProb, 70,
#                                   0.00005, 0.5, 0, 1, 0,
#                                   1, 0)

IDD = np.sum(finalDose, axis=2)
IDD = np.sum(IDD, axis=0)
x = np.arange(len(IDD))
plt.plot(x, IDD)
plt.show()
a =1 