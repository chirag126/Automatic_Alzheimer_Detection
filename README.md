#==============================================================================
#==============================================================================
#   AUTOMATIC DIAGNOSIS OF DEMENTIA USING LINGUISTIC FEATURES
#==============================================================================
#==============================================================================

These scripts were used to perform dementia classification or MMSE prediction
on the Pitts corpus (DementiaBank) data for Cookie Theft Description task.
The discourse between investigator and subject were transcribed using CHAT protocol.
Given below are the steps to extract linguistic features and perform dementia 
classification / MMSE prediction tasks.

#==============================================================================
# 1. Run data_process.py and data_process_withPauses.py
#==============================================================================

This script is used to perform pre-processing on the original transcripts.
We removed all the CHAT tags, retaining only the subject dialogues using data_process.py script.
Pauses were also retained using the data_process_withPauses.py script for obtaining utterance information.
Transcripts should be saved in "Control" and "Dementia" folders having the same path as the scripts.
When performing MMSE prediction, uncomment the 'if' statements so as to only write data with MMSE information.

#==============================================================================
# 2. Run feature_extract_new.py
#==============================================================================

This script is used to extract the features from the pre-processed metadata from the dataset.
Takes the path of two metadata csv as input arguments:
    --file_path - csv filepath for pre-processed metadata
    --file_path2 - csv filepath for preprocessesd metadata with pauses to calculate Mean Length Utterance (MLU)                

#==============================================================================
# 3. Run model.py 
#==============================================================================
Run model.py
The script takes three arguments:    
    --model - pertaining to various models (Logistic Regression("LR"), Decision Trees("DT"),
                 Random Forest("RF"), and Support Vector Machines("SVM"))    
    --type -  represents the two classification problems, viz. 
              "DEM" - Dementia classification
              "MMSE" - MMSE categorical classification    
    --file_path - takes the feature_set csv file as input
        
#==============================================================================
# For data analysis, use word_cloud_gen.py
#==============================================================================

This script is used to generate the word clouds for Control vs Dementia subjects.

# Automatic_Alzheier_Detection
