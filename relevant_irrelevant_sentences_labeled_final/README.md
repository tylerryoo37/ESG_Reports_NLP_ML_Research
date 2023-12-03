This directory holds the relevant csv files of extracted sentences from pdf reports (handled duplicates and indexed each sentence for future use):

1) The extraction process: extracted all the highlighted sentences as 'total_relevant' and all the sentences from pdfs as 'total_all'


2) Additional steps were needed to separate relevant and irrelevant sentences from 'total_all'


- <ins>string_matched.csv</ins>: matched 'total_relevant' sentences with corresponding relevant sentences that reside in 'total_all'

- <ins>sentences_not_matched.csv</ins>: keeping track of 96 instances of sentences that weren't matched

- <ins>sentences_matched.csv</ins>: keeping track of sentences that were matched 

3) Used sentences_matched file to separate relevant and irrelevant sentences from all. Also resolved ordering issue I observed earlier and made sure all duplicates are handled


4) Made sure to index all the sentences for future use


- <ins>all_sentences.csv</ins>: combined both relevant and irrelevant sentences --> total sentences = 77,277

- <ins>extracted_relevant_sentences.csv</ins>: relevant sentences with index --> irrelevant sentences = 76,430

- <ins>extracted_irrelevant_sentences.csv</ins>: irrelevant sentences with index --> relevant sentences = 847

- <ins>total_stat_final.csv</ins>: statistics of all the relevant and irrelevant sentences (percentages in relevant / total)