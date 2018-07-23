#-*- coding:utf-8 -*- 
"""
Created on Thu Jul 12 15:35:50 2018

@author: JQ
"""

import os
import codecs 
import pickle


def Lexicon(corpus_path):
    s_file_list=[corpus_path+os.sep+type_path+os.sep+type_file \
                    for type_path in os.listdir(corpus_path) \
                    for type_file in os.listdir(corpus_path+os.sep+type_path)]
    chars={}
    for s_file in s_file_list:
        with codecs.open(s_file,'rb','utf-8') as fs:
            lines=fs.readlines()
            for line in lines:
                pieces=line.strip().split()
                for piece in pieces:
                    for char in piece:
                        chars[char]=chars.get(char,0)+1
    sorted_chars=sorted(chars.items(),key=lambda x:x[1],reverse=True)
    lexicon=dict([(item[0],index+1) for index,item in enumerate(sorted_chars)])
    lexicon_reverse=dict([(index+1,item[0]) for index,item in enumerate(sorted_chars)])
    pickle.dump([lexicon,lexicon_reverse],open('lexicon.pkl','wb'))
    return lexicon,lexicon_reverse


if __name__ == '__main__':

    Lexicon('corpus')



