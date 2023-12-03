This directory holds the code (in ipynb notebooks) and results (csv) from extracting data from raw pdf files (see annoted_sustainability_reports). 

Since there were multiple iterations during this process, there are multiple versions of code notebooks and csv files here. The older ones are archived in folders labeled e.g. v1, v2, etc. while the latest version is marked with the word "final".

Notes on V5: 
- This version of the notebook attempts to use similar methods to extract both relevant and irrelevant sentences. However, when we conduct string matching, relevant_sentences are not parsed properly from all_sentences, as all_sentences are not divided into smaller chunks. Hence, we have a lot of cases where two relevant sentences are combined into one. (Refer to the V5 Slide Deck for example cases)

Notes on V6 (The Final Version of the Notebook):
- This version attempts to improve the performance of V5 by using pdftotext package to retrieve all_sentences. Now, the challenge is to make sure we can successfully conduct string matching to accurately map relevant_sentences from all_sentences. This became more difficult in V6 because we've implemented two different methods to extract relevant_sentences and all_sentences. 

- To improve string matching: we've implemented an algorithm as the following: 

Revised String Matching Algorithm: 
1) use process.extract with scorer = fuzz.WRatio w/ cutoff = 90
2) use process.extract with scorer = fuzz.partial_ratio w/ cutoff = 90
   Test Case 1: if 1) and 2) only has one extracted value compare their lengths and append the longer version
   Test Case 2: if 1) and 2) have different length of extracted sentences --> check if they have the same scorer value or not 
    a. If they do have the same scorer value: join all the sentences into one
    b. If they don't have the same scorer value: only get the max ratio 
    c. Compare these two values' length and append the one with greater length 
    
We were able to reduce the number of strings not matched from 48 to 25

The next step is to conduct further testings to verify whether V6 can be used to provide a better version of extracted sentences