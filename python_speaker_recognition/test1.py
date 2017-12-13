#!/usr/bin/env python
import numpy
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


(rate,sig) = wav.read("./wav/1/0_1.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
dd_mfcc_feat = delta(d_mfcc_feat, 2)

result=numpy.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
result_1=numpy.concatenate((result,dd_mfcc_feat),axis=1)




print(result_1)
print(result_1.shape)

print(type(result_1))
# (139, 39)
# <class 'numpy.ndarray'>