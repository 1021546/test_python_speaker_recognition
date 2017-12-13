#!/usr/bin/env python
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn import mixture

temp_train=np.empty(shape=[0, 39])

for k in range(1,7):
	for i in range(0,6):
		for j in range(1,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_"+str(j)+".wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			temp_train=np.vstack((temp_train,result_1[0:99]))




# print(temp_train)
print(temp_train.shape)
print(type(temp_train))

# print(temp_train[0])
print(temp_train[0].shape)
print(temp_train.shape[0])
print(temp_train.shape[1])


output_line = ""

for i in range(0,temp_train.shape[0]):
	target_line = ""
	for j in range(0,temp_train.shape[1]):
		target_line+=str(j+1)+":"+str(temp_train[i][j])+" "
	target_line=str(i//2970)+" "+target_line
	output_line+=target_line+"\n"


train_file = open("./train_3.txt", "w")
train_file.write(output_line)
train_file.close()


x_test=np.empty(shape=[0, 39])


for k in range(1,7):
	for i in range(0,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_3.wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			x_test=np.vstack((x_test,result_1[0:99]))

print(x_test)
print(x_test.shape)
print(type(x_test))

output_line = ""

for i in range(0,x_test.shape[0]):
	target_line = ""
	for j in range(0,x_test.shape[1]):
		target_line+=str(j+1)+":"+str(x_test[i][j])+" "
	target_line=str(i//594)+" "+target_line
	output_line+=target_line+"\n"


test_file = open("./test_3.txt", "w")
test_file.write(output_line)
test_file.close()