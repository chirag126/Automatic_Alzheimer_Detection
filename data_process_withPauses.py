#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:33:16 2017
@author: chirag212
@author: nivi_k (modified data_process for MLU calculation)

"""

#==============================================================================
# Pre-Processing the data - keeping the pauses information for MLU computing
#==============================================================================

import re 
import os
import argparse
import numpy as np
import pandas as pd

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
    string = string.replace('(.)', '1pause1.')
    count_2 = string.count("(..)")
    string = string.replace('(..)', '2pause2.')
    count_3 = string.count("(...)")
    string = string.replace('(...)', '3pause3.')
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
         
    parser.add_argument('--file_path', default=os.getcwd(), type=str,
                        help='filepath for Control and Dementia folders')
#    parser.add_argument('--lr', default=0.0663, type=int,
#                        help='learning rate')
#    parser.add_argument('--epochs', default=100, type=int,
#                        help='number of epochs')

    args = parser.parse_args()
    
    # Filenames
    control_path = os.path.join(args.file_path, 'Control', 'cookie')
    dementia_path = os.path.join(args.file_path, 'Dementia', 'cookie')
    control_list = os.listdir(control_path)
    dementia_list = os.listdir(dementia_path)

    # MetaData
    idx = 0
    temp = []
    index = range(len(control_list) + len(dementia_list))
    columns = ['filepath', 'age', 'gender', 'mmse', 'pause1','pause2', 'pause3', 
               'count_unintelligible', 'count_trailing', 'count_repetitions',  
               'category', 'data']
    
    metadata = pd.DataFrame(index=index, columns=columns)
    for file in control_list:
         with open(os.path.join(control_path, file),encoding="utf8") as f:
             content = f.read().splitlines()
         
         dialogue, count_pause, count_misc = process_string(content)
         
         for s in dialogue.split(' '):
             if s.startswith("&"):
                 temp.append(s)
                 
         age = content[5].split(':')[1].split('|')[3][:-1]
         gender = content[5].split(':')[1].split('|')[4]
         MMSE = content[5].split(':')[1].split('|')[8]
         category = content[5].split(':')[1].split('|')[5]
         metadata.loc[idx] = [file, age, gender, \
                             MMSE, count_pause[0], count_pause[1], count_pause[2], 
                             count_misc[0], count_misc[1], 
                             count_misc[2], category, dialogue]
         idx += 1
         
    for file in dementia_list:
         with open(os.path.join(dementia_path, file),encoding="utf8") as f:
             content = f.read().splitlines()
             
         dialogue, count_pause, count_misc = process_string(content)

         for s in dialogue.split(' '):
             if s.startswith("&"):
                 temp.append(s)
                 
         age = content[5].split(':')[1].split('|')[3][:-1]
         gender = content[5].split(':')[1].split('|')[4]
         MMSE = content[5].split(':')[1].split('|')[8]
         category = content[5].split(':')[1].split('|')[5]
         metadata.loc[idx] = [file, age, gender, \
                             MMSE,  count_pause[0], count_pause[1], count_pause[2], 
                             count_misc[0], count_misc[1], 
                             count_misc[2], category, dialogue]
         idx += 1
         
    metadata.to_csv(os.path.join(args.file_path, 'metadata_withPauses.csv'), index=False, encoding='utf-8')   
    return np.unique(temp)
if __name__ == '__main__':
    x = main()