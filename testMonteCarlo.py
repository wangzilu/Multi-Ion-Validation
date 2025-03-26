import os
import numpy as np
print(os.getcwd())
from protonMC_CUDA import cudaProtonMonteCarlo
beamParam         = np.load('./input/beamsparam.npy')
beamParam[0][0]   = 100000
beamParam[0][4:9] = 0
beamParam[0][3]   = 1200

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
tempSumDose = np.zeros((dims[0] * dims[1] * dims[2] * 3), dtype=np.float32)
tempLET     = np.zeros((dims[0] * dims[1] * dims[2] * 4), dtype=np.float32)
vSAD        = 20.0
a = cudaProtonMonteCarlo(finalDose, tempSumDose, tempLET, './mc_config', beamParam, sourcePos, bmdir, ctdata, corner, resolution, dims, isocenter, materialComposition, HUDensity, eneProb, vSAD, 1, 10)

a =1 