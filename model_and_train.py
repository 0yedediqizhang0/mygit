#-*- coding:utf-8 -*- 
"""
Created on Sun Jul 15 10:31:15 2018

@author: JQ
"""


from __future__ import print_function

import gensim
import os
import numpy as np
np.random.seed(1111)

from keras import regularizers
from keras.layers import Input,Embedding,Bidirectional,LSTM,Dropout,Conv1D,TimeDistributed,Dense,merge,ZeroPadding1D
from keras.models import Model

from keras_contrib.layers import CRF

from keras.utils import plot_model
from keras.callbacks import EarlyStopping


from hyperopt import STATUS_OK,Trials,tpe
from hyperas.distributions import choice,uniform
from hyperas import optim

from lexicon import Lexicon
from data import data
#使用Hyperas：密集网络
def Embedding_weights(corpus_path,lexicon_reverse):
	embedding_model=gensim.models.Word2Vec.load(r'model_conll_law.m')
	embedding_vector_size=embedding_model.vector_size
	print('词向量维数：',embedding_vector_size)
    
	embedding_weights=np.zeros((len(lexicon_reverse)+2,embedding_vector_size))
	for i in range(len(lexicon_reverse)):
		embedding_weights[i+1]=embedding_model[lexicon_reverse[i+1]]
	embedding_weights[-1]=np.random.uniform(-1,1,embedding_vector_size)
	return embedding_weights

def Bilstm_CNN_Crf(train_data,train_label,val_data,val_label,maxlen,char_value_dict_len,class_label_count,base_model_weight,nb_epoch,embedding_weights=None,is_train=True):
	word_input=Input(shape=(maxlen,),dtype='int32',name='word_input')
	if is_train:
		word_emb=Embedding(char_value_dict_len+2,output_dim=100,\
					input_length=maxlen,weights=[embedding_weights],\
					name='word_emb')(word_input)

	else:
		word_emb=Embedding(char_value_dict_len+2,output_dim=100,\
					input_length=maxlen,\
					name='word_emb')(word_input)
	
	# bilstm
	bilstm=Bidirectional(LSTM(64,return_sequences=True))(word_emb)
	bilstm_d=Dropout({{uniform(0,1)}})(bilstm)

	# cnn
	half_window_size=2
	padding_layer=ZeroPadding1D(padding=half_window_size)(word_emb)
	conv=Conv1D(nb_filter=50,filter_length=2*half_window_size+1,\
			padding='valid',activation={{choice(['relu', 'sigmoid','None'])}})(padding_layer)
	conv_d=Dropout(rate={{uniform('rate',0,1)}})(conv)
	dense_conv=TimeDistributed(Dense(50))(conv_d)

	# merge
	rnn_cnn_merge=merge([bilstm_d,dense_conv],mode='concat',concat_axis=2)
	dense=TimeDistributed(Dense(class_label_count,kernel_regularizer=regularizers.l2({{uniform(0, 1)}})))(rnn_cnn_merge)


	# crf完成分词
	crf=CRF(class_label_count,sparse_target=False)
	crf_output=crf(dense)

	# build model
	model=Model(input=[word_input],output=[crf_output])
    
	model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])
    

	print(model.input_shape)
	print(model.output_shape)

	plot_model(model, to_file='bilstm_cnn_crf_model.png',show_shapes=True,show_layer_names=True)

	if base_model_weight!=None and os.path.exists(base_model_weight)==True:
		model.load_weights(base_model_weight)

#	early_stopping=EarlyStopping(monitor='acc', patience=15, verbose=1, mode='max')

	model.fit(train_data,train_label,batch_size=256,epochs=nb_epoch,verbose=1,\
					callbacks=[EarlyStopping(monitor='val_acc',verbose=1,mode='max')],\
					validation_data=(val_data,val_label))
#    TensorBoard('./logs')
#	with open('history.txt','w') as f:
#		f.write(str(hist.history))

#	model.load_weights('best_val_model.hdf5')

	loss,acc=model.evaluate(val_data,val_label,batch_size=256)#评估模型得到训练集的损失值和准确率，score包含两个信息

	# save model
	model.save_weights('train_model.hdf5')
#	model.summary()
    
	print('Val_accuracy:', acc)

	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def main():

#	corpus_path='corpus'
#	base_model_weight='train_model.hdf5'
#	nb_epoch=4
#	label_2_index={'Pad':0,'B':1,'M':2,'E':3,'S':4,'Unk':5}
#	lexicon,lexicon_reverse=Lexicon(corpus_path)
#	print('字典大小:',len(lexicon))
#    
#	train_data,train_label,val_data,val_label=data(corpus_path,lexicon)
#	max_len=len(train_data[0])
#	print('验证集data:',val_data.shape)
#	print('验证集label:',val_label.shape)
#    
#	embedding_weights=Embedding_weights(corpus_path,lexicon_reverse)
	best_run,best_model=optim.minimize(model =Bilstm_CNN_Crf,data = data,algo =tpe.suggest,max_evals =40,\
                                     trials = Trials(),notebook_name='model_and_train')
	print(best_run)
if __name__ == '__main__':
	main()

