#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 2 12:02:41 2017
@ author: nivi_k, chirag212

"""

#==============================================================================
#   Word cloud for Control vs Dementia subjects 
#==============================================================================

import os
import nltk
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer 
#from skfeature.function.information_theoretical_based import MRMR
lmtzr = WordNetLemmatizer()

def cookie_wordcloud(data):    
    
    text_control = ""
    text_dementia = ""

    # ---------------------tagging information -------------------
    for i in range(data.shape[0]):
        content = data.loc[i]['data']        
        text = nltk.word_tokenize(content)            
        
        #  ------- POS tagging ------- 
        tag_info = np.array(nltk.pos_tag(text))
        tag_fd = nltk.FreqDist(tag for i, (word, tag) in enumerate(tag_info))

        if data.loc[i]['category'] == 'Control':            
            text_control = text_control + content
        else:
            text_dementia = text_dementia + content
  
        
    return text_control, text_dementia         

def main():
    # ------- Input data from metadata.csv ------- 
    parser = argparse.ArgumentParser(description='Processing Dementia data')
    parser.add_argument('--file_path', default=os.path.join(os.getcwd(),'metadata_allsubjects.csv'), type=str,
                        help='filepath for Control and Dementia folders')    
    args = parser.parse_args()
    data = pd.read_csv(args.file_path, encoding='utf-8')
    
    
    # ------- Feature extraction ------- 
    text_control, text_dementia = cookie_wordcloud(data)

    wordcloud_control = WordCloud(
                      relative_scaling = 1.0, collocations = False, max_words = 75, 
                      stopwords = {'to', 'of', 'is', 'in','it','that','and','there','on','for','or','the'} # set or space-separated string
                      ).generate(text_control)
    plt.figure(1)
    plt.imshow(wordcloud_control)
    plt.axis("off")
    plt.show()
    plt.title('Control word cloud')
    
    text_c = nltk.word_tokenize(text_control)    
    tag_info_c = np.array(nltk.pos_tag(text_c))
    tags_c = ' '.join(tag_info_c[:,1])

    tagcloud_control = WordCloud(
                      relative_scaling = 1.0,collocations = False, max_words = 75,                      
                      ).generate(tags_c)
    plt.figure(2)
    plt.imshow(tagcloud_control)
    plt.axis("off")
    plt.show()
    plt.title('Control tags')
 
    
    wordcloud_dementia = WordCloud(
                      relative_scaling = 1.0,collocations = False, max_words = 75, 
                      stopwords = {'to', 'of', 'is', 'in','it','that','and','there','on','for','or','the'} # set or space-separated string
                      ).generate(text_dementia)
    plt.figure(3)
    plt.imshow(wordcloud_dementia)
    plt.axis("off")
    plt.show()
    plt.title('Dementia word cloud')
    
    text_d = nltk.word_tokenize(text_dementia)    
    tag_info_d = np.array(nltk.pos_tag(text_d))
    tags_d = ' '.join(tag_info_d[:,1])
    
    tagcloud_dementia = WordCloud(
                      relative_scaling = 1.0,collocations = False, max_words = 75,                      
                      ).generate(tags_d)
    plt.figure(4)
    plt.imshow(tagcloud_dementia)
    plt.axis("off")
    plt.show()
    plt.title('Dementia tags')
 
    
    return text_control, text_dementia

if __name__ == '__main__':
    text_control, text_dementia = main()