#-*- coding:utf-8 -*- 
"""
Created on Thu Jul 12 16:07:00 2018

@author: JQ
"""

import os
import codecs 

import numpy as np
np.random.seed(1111)

from keras.utils import np_utils
from keras.preprocessing import sequence


from sklearn.cross_validation import train_test_split
import pickle

class Documents():
    def __init__(self,chars,labels,index):
        self.chars=chars
        self.labels=labels
        self.index=index

# BMES标注
def BMES(s_file_list,t_file):
    ft=codecs.open(t_file,'wb','utf-8')
    for s_file in s_file_list:
        with codecs.open(s_file,'rb','utf-8') as fs:
            lines=fs.readlines()
            for line in lines:
                word_list=line.strip().\
					replace("，"," ，").replace('。',' 。 ').replace(':',' ：').\
					replace(';',' ;').replace('、',' 、').replace('！',' ！ ').\
					replace('……',' ……').replace('？',' ？ ').replace('———',' ——— ').\
					replace('·',' · ').replace('','').replace('“','“ ').replace('”','” ').\
					replace('(','( ').replace(')',' )').replace('《','《 ').replace('》',' 》').split()
                for word in word_list:
                    if len(word)==1:         
                        ft.write(word+'\tS\n')
                    else:
                        ft.write(word[0]+'\tB\n')
                        for w in word[1:-1]:
                            ft.write(w+'\tM\n')
                        ft.write(word[-1]+'\tE\n')
                ft.write('\n')
    ft.close()

# 按句子划分dict形式保存chars，labels
def create_documents(file_name):
    documents=[]
    chars,labels=[],[]

    with codecs.open(file_name,'rb','utf-8') as f:
        index=0
        for line in f:

            line=line.strip()
            if len(line)==0:
                if len(chars)!=0:
                    documents.append(Documents(chars,labels,index))
                    chars=[]
                    labels=[]
                index+=1

            else:
                pieces=line.strip().split()
                chars.append(pieces[0])
                labels.append(pieces[1])

                if pieces[0] in ['。','，','；']:
                    documents.append(Documents(chars,labels,index))
                    chars=[]
                    labels=[]

        if len(chars)!=0:
            documents.append(Documents(chars,labels,index))
            chars,labels=[],[]
    return documents

#将chars，labels转换为数字表示
def create_matrix(documents,lexicon,label_2_index):
    data_list=[]
    label_list=[]
    index_list=[]
    for doc in documents:
        data_tmp=[]
        label_tmp=[]

        for char,label in zip(doc.chars,doc.labels):
            data_tmp.append(lexicon[char])
            label_tmp.append(label_2_index[label])

        data_list.append(data_tmp)
        label_list.append(label_tmp)
        index_list.append(doc.index)

    return data_list,label_list,index_list

#补零
def padding_sentences(data_list,label_list,max_len):
    padding_data_list=sequence.pad_sequences(data_list,maxlen=max_len)
    padding_label_list=[]
    for item in label_list:
        padding_label_list.append([0]*(max_len-len(item))+item)

    return padding_data_list,np.array(padding_label_list)


def data(corpus_path,lexicon):
    train_file=[corpus_path+os.sep+type_path+os.sep+type_file \
                for type_path in os.listdir(corpus_path) \
                for type_file in os.listdir(corpus_path+os.sep+type_path)]
    BMES(train_file,'train.data')
    
    train_documents=create_documents('train.data')
    
    label_2_index={'Pad':0,'B':1,'M':2,'E':3,'S':4,'Unk':5}
    
    train_data_list,train_label_list,train_index_list=create_matrix(train_documents,lexicon,label_2_index)
    print('句子批次数：',len(train_data_list))
    print(train_data_list[0])
    print(train_label_list[0])
    
    max_len=max(map(len,train_data_list))
    print('maxlen:',max_len)
    
    train_data,train_label_list_padding=padding_sentences(train_data_list,train_label_list,max_len)
    print('输入数据：',train_data.shape)
    print(train_data[0])

    train_label=np_utils.to_categorical(train_label_list_padding,len(label_2_index)).\
                        reshape((len(train_label_list_padding),len(train_label_list_padding[0]),-1))
    print('数据标签：',train_label.shape)
    print(train_label[0])
    
    train_data,val_data,train_label,val_label=train_test_split(train_data,train_label,test_size=0.1, random_state=0)
    return train_data,train_label,val_data,val_label

if __name__ == '__main__':
    lexicon,lexicon_reverse=pickle.load(open('lexicon.pkl','rb'))

    data('corpus',lexicon)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    