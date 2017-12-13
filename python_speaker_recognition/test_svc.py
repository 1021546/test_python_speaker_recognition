#!/usr/bin/env python
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn import mixture
import pdb # Debugger Command

x_train=np.empty(shape=[0, 39])

for k in range(1,7):
	for i in range(0,6):
		for j in range(1,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_"+str(j)+".wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			x_train=np.vstack((x_train,result_1[0:99]))




# print(x_train)
# print(x_train.shape)
# print(type(x_train))

# print(x_train[0])
# print(x_train[0].shape)
# print(x_train.shape[0])
# print(x_train.shape[1])


x_test=np.empty(shape=[0, 39])


for k in range(1,7):
	for i in range(0,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_4.wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			x_test=np.vstack((x_test,result_1[0:99]))

# print(x_test)
# print(x_test.shape)
# print(type(x_test))


y_train=np.zeros(17820, dtype=np.int)

index=0
for i in range(0,6):
	y_train[index:(index+2970)]=i
	index+=2970

# y_train[0:2970]=0
# y_train[2970:5940]=1
# y_train[5940:8910]=2
# y_train[8910:11880]=3
# y_train[11880:14850]=4
# y_train[14850:17820]=5

y_test=np.zeros(3564, dtype=np.int)

index=0
for i in range(0,6):
	y_test[index:(index+594)]=i
	index+=594

# y_test[0:594]=0
# y_test[594:1188]=1
# y_test[1188:1782]=2
# y_test[1782:2376]=3
# y_test[2376:2970]=4
# y_test[2970:3564]=5

# Import the `svm` model
from sklearn import svm

# Create the SVC model 
svc_model = svm.SVC(gamma=0.5, C=32., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(x_train, y_train)
# Apply the classifier to the test data, and view the accuracy score
print(svc_model.score(x_test, y_test))

# Predict the label of `X_test`
print(svc_model.predict(x_test))

# Print `y_test` to check the results
print(y_test)