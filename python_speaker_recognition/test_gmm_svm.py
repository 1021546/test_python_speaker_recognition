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
# print(temp_train.shape)

# print(type(temp_train))

# print(temp_train[0])
# print(temp_train[0].shape)




temp_gmm = np.empty(shape=[0, 39])
gmm_mean=np.empty(shape=[0, 39])

for i in range(0,6):
	for j in range(0,30):
		temp_gmm=np.vstack((temp_gmm,temp_train[i*30+j]))
		# print(temp_gmm)
		# print(temp_gmm.shape)
	# fit a Gaussian Mixture Model with two components
	clf = mixture.GaussianMixture(n_components=6, covariance_type='full')
	clf.fit(temp_gmm)
	# print(clf.means_)
	# print(clf.means_.shape)
	gmm_mean=np.vstack((gmm_mean,clf.means_))
	temp_gmm = np.empty(shape=[0, 39])

print(gmm_mean)
print(gmm_mean.shape)

x_train=gmm_mean


x_test=np.empty(shape=[0, 39])


for k in range(1,7):
	for i in range(0,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_5.wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=np.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=np.concatenate((result,dd_mfcc_feat),axis=1)

			x_test=np.vstack((x_test,result_1[0:99]))

print(x_test)
print(x_test.shape)
print(type(x_test))



y_train=np.zeros(36, dtype=np.int)

index=0
for i in range(0,6):
	y_train[index:(index+6)]=i
	index+=6

# y_train[0:6]=0
# y_train[6:12]=1
# y_train[12:18]=2
# y_train[18:24]=3
# y_train[24:30]=4
# y_train[30:36]=5

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