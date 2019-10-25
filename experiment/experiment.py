#!/usr/bin/env python3

import os

from data import Data, DataRandom
from target_generator import TGAlpha
from damager import DMGNA, DMGNoiseFeatures
from fixer import FXZero, FXMean
import base64

cwd = os.getcwd()

try:
    os.system(f"cd {cwd} & rm data/* & rm plots/*")
except:
    pass


# create data

features = 15
observations = 10000
noiseFeatures = 5
split = 0.3
na = 0.001

damagedData = (
    DataRandom(features=features, observations=observations)
    .makeTarget(TGAlpha())
    .damage(DMGNoiseFeatures(), noiseFeatures)
    .split(testSize=split)
    .damage(DMGNA(), na)
    .save("data")
)

fixZeroData = Data().read(url=damagedData).fix(FXZero()).save("data")
fixMeanData = Data().read(url=damagedData).fix(FXMean()).save("data")

dataSignatureFixZero = (
    f"F={features}, OBS={observations}, NF={noiseFeatures}, NA={na}, FX=0, SPL={split}"
)
dataSignatureFixMean = (
    f"F={features}, OBS={observations}, NF={noiseFeatures}, NA={na}, FX=Mu, SPL={split}"
)


cmd = f'cd {cwd} & ./fit-data-and-plot.py --data {fixZeroData} --signature "{dataSignatureFixZero}" & ./fit-data-and-plot.py --data {fixMeanData} --signature "{dataSignatureFixMean}" '

os.system(cmd)
