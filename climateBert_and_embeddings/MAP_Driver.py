#!/user/as6430/.conda/envs/nlp/bin/python
import os
import torch
import PyPDF2
import nltk
from scipy.spatial.distance import cosine
import numpy as np
import fitz
from sklearn.neighbors import KDTree
import textdistance
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse


def driver():

    #check directory
    directory = 'annoted_sustainability_reports'
    headings = os.listdir(directory)
    res = []

    #writing the header string to Stats.txt file
    path = os.path.join(r"C:\Users\ayush\Downloads",'Stats.txt')
    with open(path,'a') as f:
        f.write(f'Company \t Total # of sentences \t # of relevant sentences \t AP \n')


    for head in headings[0:2]:
        new_dir = os.path.join(directory,head)
        reports = os.listdir(new_dir)
        for rep in reports[0:2]:
            inp = "grid_run --grid_mem=1G" + " ./MAP_Calc_Script.py " + os.path.join(new_dir,rep)
            os.system(inp)
            

if __name__ == '__main__':
    driver()