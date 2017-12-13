#!/usr/bin/env python
import numpy
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

batch_size = 128
num_classes = 6
epochs = 10


temp=[]

for k in range(1,7):
	for i in range(0,6):
		for j in range(1,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_"+str(j)+".wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=numpy.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=numpy.concatenate((result,dd_mfcc_feat),axis=1)

			temp.append(result_1[0:99])

x_train = numpy.stack(temp)

print(x_train)
print(x_train.shape)

print(type(x_train))


x_train = x_train.reshape(-1, 39)

print(x_train)
print(x_train.shape)

print(type(x_train))

temp=[]


for k in range(1,7):
	for i in range(0,6):

			(rate,sig) = wav.read("./wav/"+str(k)+"/"+str(i)+"_5.wav")
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			dd_mfcc_feat = delta(d_mfcc_feat, 2)

			result=numpy.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
			result_1=numpy.concatenate((result,dd_mfcc_feat),axis=1)

			temp.append(result_1[0:99])


x_test = numpy.stack(temp)

print(x_test)
print(x_test.shape)

print(type(x_test))

x_test = x_test.reshape(-1, 39)

print(x_test)
print(x_test.shape)

print(type(x_test))


y_train=numpy.zeros(17820, dtype=numpy.int)

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

y_test=numpy.zeros(3564, dtype=numpy.int)

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


# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model_bm = Sequential()
model_bm.add(Dense(128, input_dim=39))
model_bm.add(Activation('sigmoid'))
model_bm.add(Dense(256))
model_bm.add(Activation('sigmoid'))
model_bm.add(Dense(6))
model_bm.add(Activation('softmax'))

''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
model_bm.compile(loss= 'mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])


'''Fit models and use validation_split=0.1 '''
history_bm = model_bm.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							shuffle=True,
                    		validation_split=0.1)

score = model_bm.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


'''Access the loss and accuracy in every epoch'''
loss_bm	= history_bm.history.get('loss')
acc_bm 	= history_bm.history.get('acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(1)
plt.subplot(121)
plt.plot(range(len(loss_bm)), loss_bm,label='Sigmoid')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_bm)), acc_bm,label='Sigmoid')
plt.title('Accuracy')
plt.show()