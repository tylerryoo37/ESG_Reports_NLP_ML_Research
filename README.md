# t3

File Descriptions:

The purpose of this project is to study how companies report on their environmental impacts. To that end, this repository contains code/files that extract data from environmental reports by major corporations and analyze and model that data.

There are several folders in this repository, as listed below (for brevity, individual files are not included in the tree diagram): 

**annotated_sustainability_reports** holds the original company reports from which data was extracted. Those reports are organized by the sector/industry to which the reporting company belongs.

**classifier models** holds code notebooks that train and evaluate classifier models (e.g., logistic regression, KNN)

**climateBert_and_embeddings** holds notebooks and files related to the climateBERT model and related word and sentence embeddings

**company_names_info** holds basic information about the companies studied -- their names, industry, etc.

**extracted_sentences** holds the notebooks that extract relevant and irrelevant sentences from the company reports. Folders ending in "csv" hold the csv files listing the results of those extractions. There are multiple versions of notebooks and files, but the latest can be found in "final_sentences_csv" and "final_extracted_statistics_notebooks".

**train_test_splits** holds csv files containing the training and test split data. (Temporary: the code generating the csv files can be found in code block 14 of the notebook [here](https://github.com/arxivlab/t3/blob/main/extracted_sentences/notebooks/v1_extracted_statistics_notebooks/All_Annotated_Data.ipynb).

**web_scraper** holds the code file used to scrape company reports from the web

<img width="351" alt="Screen Shot 2022-10-10 at 11 50 22 AM" src="https://user-images.githubusercontent.com/89430042/194906861-f2369d4d-d20f-4dc2-8213-c135145ebdf1.png">

To see current results of irreleveant and relevant sentences (sorted by company), click [here](https://github.com/arxivlab/t3/blob/main/extracted_sentences/final_sentences_csv/final_stat.csv). 
