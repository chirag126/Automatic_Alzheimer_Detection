#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:21:52 2017

@author: chirag212
"""

#==============================================================================
# Pre-Processing the data
#==============================================================================

import re 
import os
import nltk
import math
from nltk.corpus import brown
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer 
lmtzr = WordNetLemmatizer()

def get_tag_info(content, data_pauses, data_misc):    
    
    # ---------- Initialize ----------
    data_tag_info = []
    feature_set = []
    ttr = {}
    
#    columns = ['filepath', 'ttr', 'ttr_l','R', 'num_concepts_mentioned', 'Category', 'ARI', 'CLI',
#               'prp_count', 'prp_noun_ratio', 'Gerund_count', 'NP_count', 'VP_count', 'VGP_count', 'word_sentence_ratio', 'MLU','count_pauses', 'count_unintelligible',
#               'count_trailing', 'count_repetitions', 'MMSE', 'SIM_score', 'Bruten']

    # ---------- Check lists ----------   
    noun_list = ['NN', 'NNS', 'NNP', 'NNPS']
    verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']
 
    cookie_pic_list = ['cookie','jar','stool','steal', \
                        'sink','basin','kitchen', \
                        'window','curtain','fall', 'take', 'tip', 'topple','outside']
    
    list1 = ['mother','woman','lady','mama','mommy']
    list2 = ['girl','daughter','sister']
    list3 = ['boy','son','child','kid','brother']
    list4 = ['dish','plate','cup']
    list5 = ['overflow','spill','running']
    list6 = ['dry','wash', 'wipe', 'clean']
    list7 = ['tap', 'spigot', 'spout','faucet']
    list8 = ['counter','cabinet','shelf','cupboard']
    list9 = ['water','puddle','pool']
    
    # SUBJECTS (boy, girl, mother), # PLACES (kitchen, exterior seen through the window), 
    # OBJECTS (cabinet, cookies, counter, curtain, dishes on the counter, faucet, floor, jar, plate, sink, stool, water, window) and 
    # ACTIONS (boy taking the cookie, boy or stool falling, woman drying or washing dishes/plate, 
    # water overflowing or spilling, the girl asking for a cookie, 
    # woman unconcerned by the overflowing, # woman indifferent to the children)
   
    # ---------- Define production rules / VP, NP definition ----------
    grammar = r"""
    DTR: {<DT><DT>}
    NP: {<DT>?<JJ>*<NN.*>} 
    PP: {<IN><NP>} 
    VPG: {<VBG><NP | PP>}
    VP: {<V.*><NP | PP>}     
    CLAUSE: {<NP><VP>} 
    """  
    
    # ---------------------tagging information -------------------
    content_pauses = data_pauses
    text = nltk.word_tokenize(content)    
    
    # ========= LEXICOSYNTACTIC FEATURES =========
    
    #  ------- POS tagging ------- 
    tag_info = np.array(nltk.pos_tag(text))
    tag_fd = nltk.FreqDist(tag for i, (word, tag) in enumerate(tag_info))
    freq_tag = tag_fd.most_common()
    data_tag_info.append(freq_tag)
    
    # ------- Lemmatize each word -------    
    text_root = [lmtzr.lemmatize(j) for indexj, j in enumerate(text)]
    for indexj, j in enumerate(text):
        if tag_info[indexj,1] in noun_list:
            text_root[indexj] = lmtzr.lemmatize(j) 
        elif tag_info[indexj,1] in verb_list:
            text_root[indexj] = lmtzr.lemmatize(j,'v')             
    
    # ------- Phrase type ------- 
    sentence = nltk.pos_tag(text)
    cp = nltk.RegexpParser(grammar)
    phrase_type = cp.parse(sentence)  
    
    # ------- Pronoun frequency -------
    prp_count = sum([pos[1] for pos in freq_tag if pos[0]=='PRP' or pos[0]=='PRP$'])
    
    # ------- Noun frequency -------
    noun_count = sum([pos[1] for pos in freq_tag if pos[0] in noun_list])
    
    # ------- Gerund frequency -------
    vg_count = sum([pos[1] for pos in freq_tag if pos[0]=='VBG'])
    
    # ------- Pronoun-to-Noun ratio -------
    if noun_count != 0:
        prp_noun_ratio = prp_count/noun_count
    else:
        prp_noun_ratio = prp_count
    
    # Noun phrase, Verb phrase, Verb gerund phrase frequency        
    NP_count = 0
    VP_count = 0
    VGP_count = 0
    for index_t, t in enumerate(phrase_type):
        if not isinstance(phrase_type[index_t],tuple):
            if phrase_type[index_t].label() == 'NP':
                NP_count = NP_count + 1;
            elif phrase_type[index_t].label() == 'VP': 
                VP_count = VP_count + 1;
            elif phrase_type[index_t].label() == 'VGP':
                VGP_count = VGP_count + 1;
                        
    # ------- TTR type-to-token ratio ------- 
    numtokens = len(text)
    freq_token_type = Counter(text)  # or len(set(text)) # text_root
    v = len(freq_token_type)
    ttr = float(v)/numtokens       
               
    # ------- TTR type-to-token ratio lemmatized------- 
    freq_lemmtoken_type = Counter(text_root)  # or len(set(text)) # text_root
    vl = len(freq_lemmtoken_type)
    ttr_lemmatized = float(vl)/numtokens                         
    
    # ------- Honore's statistic ------- 
    freq_token_root = Counter(text_root)
    occur_once = 0
    for j in freq_token_root:
        if freq_token_root[j] == 1:
            occur_once = occur_once + 1
    v1 = occur_once
    R = 100 * math.log(numtokens / (1 - (v1/v)))
            
    # ------- Automated readability index ------- 
    num_char = len([c for c in content if c.isdigit() or c.isalpha()])
    num_words = len([word for word in content.split(' ') if not word=='' and not word=='.'])
    num_sentences = content.count('.') + content.count('?')
    ARI = 4.71*(num_char/num_words) + 0.5*(num_words/num_sentences) - 21.43
    
    # ------- Coleman–Liau index -------
    L = (num_char/num_words)*100
    S = (num_sentences/num_words)*100
    CLI = 0.0588*L - 0.296*S - 15.8                
        
    # ------- word_sentence_ratio -------
    word_sentence_ratio = num_words/num_sentences
    
    # ========= SEMANTIC FEATURES =========
    
    # ------- Mention of concept ------- 
    num_concepts_mentioned = len(set(cookie_pic_list) & set(freq_token_root)) \
                            + len(set(list1) & set(freq_token_root)) + len(set(list2) & set(freq_token_root)) \
                            + len(set(list3) & set(freq_token_root)) + len(set(list4) & set(freq_token_root)) \
                            + len(set(list5) & set(freq_token_root)) + len(set(list6) & set(freq_token_root)) \
                            + len(set(list7) & set(freq_token_root)) + len(set(list8) & set(freq_token_root)) \
                            + len(set(list9) & set(freq_token_root))           								
    
    # ========= ACOUSTIC FEATURES =========
    
    # ------- Pauses and unintelligible count -------
    count_pauses = data_pauses[1] + data_pauses[2]
    
    count_unintelligible = data_misc[0]
    
    count_trailing = data_misc[1]
    
    count_repetitions = data_misc[2]
    
    # ---------- Bruten Index ----------    
    bruten = float(vl)**(numtokens**-0.0165)
   
    feature_set = [ttr, R, num_concepts_mentioned,
                   ARI, CLI, prp_count, prp_noun_ratio, 
                   vg_count, NP_count, VP_count, VGP_count,
                   word_sentence_ratio, count_pauses, count_unintelligible,
                   count_trailing, count_repetitions, bruten]
        
    return feature_set  

def process_string(content):    
    # Processing strings
    string = ''
    flag = 0
    for f in content:
        if f.startswith('%mor') or f.startswith('*INV'):
            flag = 0
        if f.startswith('*PAR:'):
            flag = 1
            string = string + f[4:]
        if flag == 1 and (not f.startswith('*PAR')):
            string = string + f
#        print (string)
            
    string = ''.join(i for i in string if not i.isdigit())

    # COUNT FILLERS ================================================================

    # Count of trailing
    count_trailing = string.count('+...')
    
    # Count pauses
    count_pause = []
    count_1 = string.count("(.)")
    string = string.replace('(.)', '')
    count_2 = string.count("(..)")
    string = string.replace('(..)', '')
    count_3 = string.count("(...)")
    string = string.replace('(...)', '')
    count_pause = [count_1, count_2, count_3]
    
    # Count of unintelligible words
    count_unintelligible = string.count('xxx')
    
    # Count of repetitions
    count_repetitions = string.count('[/]')
    
    count_misc = [count_unintelligible, count_trailing, count_repetitions]
    
    # REMOVE EXTRA TAGS and FILLERS================================================
    
    string = string.replace("\t", ' ')
    #==============================================================================
    #     Group 1
    #==============================================================================
    # Remove paranthesis '()'
    string = string.replace('(', '')
    string = string.replace(')', '')

    #==============================================================================
    #     Group 2
    #==============================================================================
    # Remove paranthesis '&=clears throat'
    string = string.replace('&=clears throat', '')
    string = string.replace('&=clears:throat', '')
	    
    # Remove paranthesis '&=anything', '&anything', and '=anything'
    bad_chars = ["&=", '&', "=", "+"]
    
    for bad_str in bad_chars:
        string = ' '.join(s for s in string.split(' ') if not s.startswith(bad_str))
    
    #==============================================================================
    #     Group 3
    #==============================================================================
    # Remove paranthesis '[* anything]', '[+ anything]', [: anything], '[=anything]',
    #                    '[/anything]', '[x anything]', and'[% anything]'
    bad_chars = ["[*", "[+", "[:", "[=", "[/", "[x", "[%"]
    for bad_str in bad_chars:
        string = ' '.join(s for s in string.split(' ') if not s.startswith(bad_str) and not s.endswith(']'))
    
    #string = re.sub(r'\[.*\]', '', string)
      
    #==============================================================================
    #     Group 4, 5, 6, 7
    #==============================================================================
    bad_chars = ["+=", '<', '>', '^', "xxx", '@', " _ ", " _ :", "+//"]
    for bad_str in bad_chars:
        string = string.replace(bad_str, '')

    #==============================================================================
    #     Group 8
    #==============================================================================        
    bad_chars = ["sr-ret", "pw-ret", "sr-rep", "s:r-ret", "p:w-ret", "s:r-rep", "s:r"]
    for bad_str in bad_chars:
        string = string.replace(bad_str, '')        
    
    #==============================================================================
    #     Group 9
    #==============================================================================
    bad_chars = ["[", "]", ":","_","-","+", "*", '\x15']
    for bad_str in bad_chars:
        string = string.replace(bad_str, '')    

	#==============================================================================
    #     Group 10
    #==============================================================================
    bad_chars = ["mhm .", "hmhmm .", "hmm .", "okay .", "hm .", "alright .", "well .", "oh ."]
    for bad_str in bad_chars:
        string = string.replace(bad_str, '') 

		
    string = string.replace('  ', ' ')
    string = string.replace('..', '.')
    string = string.replace('. .', '.')
    return string, count_pause, count_misc

    
def main():
    parser = argparse.ArgumentParser(description='Processing Dementia data')
    
    parser.add_argument('--file_path', default='/home/chirag212/Nivedita/Pitt/', type=str,
                        help='filepath for Control and Dementia folders')
    
#    parser.add_argument('--file_path', default=os.getcwd(), type=str,
#                        help='filepath for Control and Dementia folders')

    args = parser.parse_args()
    control = {}
    dementia = {}
    
    # Filenames
    control_path = os.path.join(args.file_path, 'Control', 'cookie')
    dementia_path = os.path.join(args.file_path, 'Dementia', 'cookie')
    control_list = os.listdir(control_path)
    dementia_list = os.listdir(dementia_path)
    
    control_list = sorted(control_list, key=lambda x: (int(re.sub('\D','',x)),x))
    dementia_list = sorted(dementia_list, key=lambda x: (int(re.sub('\D','',x)),x))
    
    # MetaData
    for file in control_list:
         with open(os.path.join(control_path, file),encoding="utf8") as f:
             content = f.read().splitlines()
         category = content[5].split(':')[1].split('|')[5]    
         p_id = os.path.join(control_path, file).split('/')[-1].split('-')[0]
#         visit = os.path.join(control_path, file).split('/')[-1].split('.')[0][-1]
         
         dialogue, count_pause, count_misc = process_string(content)
         feature_set = get_tag_info(dialogue, count_pause, count_misc)
         feature_set.append(category)
         if p_id in control.keys():
             control[p_id].append(feature_set)
         else:
             control[p_id] = []
             control[p_id].append(feature_set)
         
         
    for file in dementia_list:
         with open(os.path.join(dementia_path, file),encoding="utf8") as f:
             content = f.read().splitlines()
         category = content[5].split(':')[1].split('|')[5]    
         p_id = os.path.join(dementia_path, file).split('/')[-1].split('-')[0]
#         visit = os.path.join(dementia_path, file).split('/')[-1].split('.')[0][-1]
         
         dialogue, count_pause, count_misc = process_string(content)
         feature_set = get_tag_info(dialogue, count_pause, count_misc)
         feature_set.append(category)
         if p_id in dementia.keys():
             dementia[p_id].append(feature_set)
         else:
             dementia[p_id] = []
             dementia[p_id].append(feature_set)
         
    
    return control, dementia

if __name__ == '__main__':
    longitudanal_control, longitudanal_dementia = main()
    
    for key in longitudanal_dementia.keys():
        category = np.array(longitudanal_dementia[key])[0, -1]
        if len(longitudanal_dementia[key]) == 5:
            plt.figure()
            plt.plot(np.array(longitudanal_dementia[key])[:, 15], '-o')
            plt.title('{}--{}'.format(key, category))
#            plt.legend(range(len(longitudanal_control[key])))