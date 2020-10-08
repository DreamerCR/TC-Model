# coding: utf-8
#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------
## import packages
import datetime
starttime = datetime.datetime.now()

import os
import keras
import numpy as np
import data_load
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import backend as K
from tensorflow.keras import datasets
from tensorflow.keras import metrics
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Convolution2D
from tensorflow.keras.layers import Dense,Concatenate,Dropout,Activation,Flatten
from tensorflow.keras.utils import multi_gpu_model

from matplotlib import pyplot
from sklearn import model_selection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-----------------------------------------------------------------------------------------------------
## Set the parameters of input datasets
area='WP' # the experimental area
timestep=5 # the timestep of LSTM module
r=6; grid_num=81-8*(10-r)  # the region range
level_num=12 # the number of pressure levels
plvar_num=5 # the number of atmospheric variables
sfvar_num=1 # the number of oceanic variables

# Set the hyper-parameters of our model
batch_size=64
epochs=25
lr=0.0003 # the learning rate

#-----------------------------------------------------------------------------------------------------
## Define the hybrid CNN-LSTM model
# Define the 3DCNN module
pl_model = Sequential()
pl_model.add(layers.TimeDistributed(layers.Conv3D(filters=32,kernel_size=(5,5,1), strides=(2,2,1),kernel_initializer=initializers.glorot_uniform(seed=42),
data_format='channels_last',activation='relu'),input_shape=(timestep,grid_num,grid_num,level_num,plvar_num)))
#pl_model.add(layers.normalization.BatchNormalization())
pl_model.add(layers.TimeDistributed(layers.Conv3D(filters=64,kernel_size=(5,5,1), strides=(2,2,1),
data_format='channels_last',activation='relu')))
#pl_model.add(layers.normalization.BatchNormalization())
pl_model.add(layers.TimeDistributed(layers.Conv3D(filters=128,kernel_size=(5,5,1), strides=(2,2,1),
data_format='channels_last',activation='relu')))
#pl_model.add(layers.normalization.BatchNormalization())
#pl_model.add(layers.TimeDistributed(layers.Conv3D(filters=256,kernel_size=(5,5,1), strides=(1,1,1),
#data_format='channels_last',activation='relu')))
#pl_model.add(layers.TimeDistributed(layers.Conv3D(filters=512,kernel_size=(3,3,2), strides=(1,1,1),
#data_format='channels_last',activation='relu')))

# Remove the extra dimension by using flatten
pl_model.add(layers.TimeDistributed(layers.Flatten()))
pl_model.add(layers.TimeDistributed(layers.Dense(100,activation='relu')))

# Define the 2DCNN module
sf_model=Sequential()
sf_model.add(layers.TimeDistributed(layers.Convolution2D(filters=32,kernel_size=(5,5), strides=(2,2),kernel_initializer=initializers.glorot_uniform(seed=42),
    data_format='channels_last',activation='relu'),input_shape=(timestep,grid_num,grid_num,sfvar_num)))
#sf_model.add(layers.normalization.BatchNormalization())
sf_model.add(layers.TimeDistributed(layers.Convolution2D(filters=64,kernel_size=(5,5), strides=(2,2),
    data_format='channels_last',activation='relu')))
#sf_model.add(layers.normalization.BatchNormalization())
sf_model.add(layers.TimeDistributed(layers.Convolution2D(filters=128,kernel_size=(5,5), strides=(2,2),
    data_format='channels_last',activation='relu')))
#sf_model.add(layers.normalization.BatchNormalization())
#sf_model.add(layers.TimeDistributed(layers.Convolution2D(filters=256,kernel_size=(5,5), strides=(1,1),
  # data_format='channels_last',activation='relu')))
#sf_model.add(layers.TimeDistributed(layers.Convolution2D(filters=512,kernel_size=(3,3),strides=(1,1),
    #data_format='channels_last',activation='relu')))

sf_model.add(layers.TimeDistributed(layers.Flatten()))
sf_model.add(layers.TimeDistributed(layers.Dense(100,activation='relu')))

merged=Merge([pl_model,sf_model],mode='concat')

# Define the LSTM module
TC_model=Sequential()
TC_model.add(merged)
TC_model.add(layers.LSTM(100))
TC_model.add(Dense(1, activation='sigmoid'))

# set the numbers of GPU
TC_model=multi_gpu_model(TC_model,gpus=2)


#-----------------------------------------------------------------------------------------------------
## Get the data and bulid datesets used to train and test the hybrid model
# Get the path of input data
pltxt_path='/Users/cr/Research/2018/WP_pl_origin.txt'
sftxt_path='/Users/cr/Research/2018/WP_sf_origin.txt'

plvar,pllabel=data_load.pl_load(pltxt_path,timestep,r)
sfvar,sflabel=data_load.sf_load(sftxt_path,timestep,r)

plvar,pllabel=np.array(plvar),np.array(pllabel)
sfvar,sflabel=np.array(sfvar),np.array(sflabel)

# Divide the dataset to train and test
plvar_train,plvar_test,sfvar_train,sfvar_test,pllabel_train,pllabel_test = model_selection.train_test_split(plvar,sfvar,pllabel,test_size=0.3,random_state=42)

#-----------------------------------------------------------------------------------------------------
## Set the evalution metric(ROC value) 
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
        self.AUC=[]
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            self.AUC.append(score)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score))

RocAuc = RocAucEvaluation(validation_data=([plvar_test,sfvar_test],pllabel_test), interval=1)

#-----------------------------------------------------------------------------------------------------
## Compile, Train and Evaluate our model
TC_model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])

#keras.backend.get_session().run(tf.global_variables_initializer())
#es = keras.callbacks.TensorBoard('./log2',histogram_freq=1,batch_size=batch_size,write_graph=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

history = TC_model.fit([plvar_train,sfvar_train], pllabel_train, shuffle = True, batch_size=batch_size, epochs=epochs, validation_data=([plvar_test,sfvar_test],pllabel_test), verbose=2)
score,acc = TC_model.evaluate([plvar_test,sfvar_test], pllabel_test)

#-----------------------------------------------------------------------------------------------------
## Plot
# Acc=history.history['val_acc']
# Auc=RocAuc.AUC
# epoch=[]
# for epoch_i in range(epochs):
#     epoch_new=epoch_i+1
#     epoch.append(epoch_new)
# #print Acc,Auc,epoch

# #plot dataset
# file_path='/home/zxy/Desktop/CR/TC_CR/result/'+str(area)+'_epochs='+str(epochs)+
# '_batchsize='+str(batch_size)+'_lr='+str(lr)+'.txt'
# file=h5py.File(file_path,'w')
# file.create_dataset('Acc',data=Acc)
# file.create_dataset('Auc',data=Auc)
# file.create_dataset('epoch',data=epoch)
# file.close()

# pyplot.plot(Acc,label='Acc')
# pyplot.plot(Auc,label='Auc')
# pyplot.title('NA_result_1')
# pyplot.ylabel('ACC')
# pyplot.xlabel('epochs')
# pyplot.legend()
# #pyplot.show()

# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.title(str(area)+'_'+'batch_size='+str(batch_size)+'_'+'epochs='+str(epochs)+'_'+
#     'lr='+str(lr)+'_'+'time='+str(time)+'_acc='+str(acc)+'_ROC='+str(Auc[epochs-1]))
# pyplot.ylabel('loss')
# pyplot.xlabel('epochs')
# pyplot.legend()
# #pyplot.show()

#-----------------------------------------------------------------------------------------------------
## Save the model
#TC_model.save('E:\CR\'+str(endtime[0:16])+'.h5')
