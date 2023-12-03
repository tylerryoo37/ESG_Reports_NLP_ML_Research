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
# def pdf_extractor(path):
#     pdfReader = PyPDF2.PdfFileReader(path)
#     n = len(pdfReader.pages)
#     print(n)
#     txt = ''
#     for i in range(n):
#         temp = pdfReader.pages[i].extractText()
#         txt = txt + temp
        
#     tokens = nltk.sent_tokenize(txt.replace('\n', ''))
#     #giving new name for preprocess strings
#     #keeping only sentences with length greater than 3
#     pre_tokens = []
#     for sen in tokens:
#         if len(sen.split(' ')) > 3:
#             pre_tokens.append(sen)

#     return pre_tokens

def kw_mean_gen(model,tokenizer):

    keywords = ['Electric vehicle',
                'Solar',
                'Wind',
                'Hydroelectric' ,
                'Nuclear',
                'REC',
                'Efficiency',
                'Deforestation',
                'Afforestation',
                'carbon',
                'credit',
                'capture',
                'sequestration',
                'storage',
                'Hydrogen',
                'Geothermal',
                'Biomass',
                'Renewable', 
                'Energy',
                'emissions',
                'reforestation',
                'Decreased', 
                'Reduced']
    
    target = []
    for ids,sen in enumerate(keywords):
        
        marked_text = "[CLS] " + sen + " [SEP]"
        # Split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(marked_text)


        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


        #creating segment ids for the sentence
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        hidden_states = []
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Evaluating the model will return a different number of objects based on 
            # how it's  configured in the `from_pretrained` call earlier. In this case, 
            # becase we set `output_hidden_states = True`, the third item will be the 
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

            hidden_states = outputs[-1]


        #print('Tensor shape for each layer: ', hidden_states[0].size())

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # token_embeddings.size()

        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            sum_vec = torch.sum(token[-4:], dim=0)

            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(sum_vec)
            
        target.append(torch.stack(token_vecs_cat, dim = 0).mean(dim = 0))

    #calculating mean of keywords of the word embeddings
    kw_mean = torch.stack(target, dim = 0).mean(dim = 0)

    return kw_mean

def Sen_Embedder(pre_tokens,model,tokenizer):

    corpus = []

    for sents in pre_tokens:
        target = []
        #splitting the sentences on ' '
        for ids,sen in enumerate(sents.split(' ')):
            
            marked_text = "[CLS] " + sen + " [SEP]"
            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)


            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


            #creating segment ids for the sentence
            segments_ids = [1] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            hidden_states = []
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
                # Evaluating the model will return a different number of objects based on 
                # how it's  configured in the `from_pretrained` call earlier. In this case, 
                # becase we set `output_hidden_states = True`, the third item will be the 
                # hidden states from all layers. See the documentation for more details:
                # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

                hidden_states = outputs[-1]


            #print('Tensor shape for each layer: ', hidden_states[0].size())

            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)

            # Swap dimensions 0 and 1.
            token_embeddings = token_embeddings.permute(1,0,2)

            # token_embeddings.size()

            # Stores the token vectors, with shape [22 x 3,072]
            token_vecs_cat = []

            # `token_embeddings` is a [22 x 12 x 768] tensor.

            # For each token in the sentence...
            for token in token_embeddings:

                # `token` is a [12 x 768] tensor

                # Concatenate the vectors (that is, append them together) from the last 
                # four layers.
                # Each layer vector is 768 values, so `cat_vec` is length 3,072.
                sum_vec = torch.sum(token[-4:], dim=0)

                # Use `cat_vec` to represent `token`.
                token_vecs_cat.append(sum_vec)

            target.append(torch.stack(token_vecs_cat, dim = 0).mean(dim = 0))
            
        corpus.append(torch.stack(target, dim = 0).mean(dim = 0))

    return corpus

def BERT_Model():

    tokenizer = AutoTokenizer.from_pretrained("climatebert/distilroberta-base-climate-f")
    model = AutoModelForMaskedLM.from_pretrained("climatebert/distilroberta-base-climate-f",output_hidden_states = True)    
    model.eval()

    return model,tokenizer

def ranking(corpus,kw_mean,pre_tokens):

    a = torch.stack(corpus)
    all_embeddings = a
    normed_embeddings = (all_embeddings.T / (all_embeddings**2).sum(axis=1) ** 0.5).T
    indexer = KDTree(normed_embeddings)

    top = indexer.query(kw_mean.reshape(1, -1),return_distance = False, k = len(pre_tokens))
    ranked = np.array(pre_tokens)[top]

    return ranked

def cal_pre(hg_text,ranked):

    result = []
    pre_list = []
    for x in hg_text:
        mx = -1
        for idx,y in enumerate(list(ranked[0])):
            score = textdistance.LCSStr().similarity(x,y)
            if score > mx:
                mx = score
                ans = idx
            
        result.append([x,ans])
        pre_list.append(ans)
    #sort pre to find the AP
    pre_list = sorted(pre_list)

    ans = 0
    for idx,ele in enumerate(pre_list):
        ele += 1
        idx += 1
        ans += (idx/ele)

    #calculate precision for all ranked sentences in doc
    cur = 0
    #getting precision for all sentences
    all_pre = []
    for idx,sen in enumerate(list(ranked[0])):
        if (cur < len(pre_list)) and (idx >= pre_list[cur]):
            cur += 1
            
        all_pre.append(cur/(idx+1))

    return (ans/len(pre_list)),pre_list, all_pre  

#Test to Extract All and Highlighted text from pdf
def hg_txt_extractor(file_path):
    #testing how to extract highlighted text
    doc = fitz.open(file_path)
    # Total page in the pdf
    # taking page for further processing
    result = []
    txt = ''
    for page in doc:
        # list to store the co-ordinates of all highlights
        highlights = []
        # loop till we have highlight annotation in the page
        annot = page.first_annot
        while annot:
            if annot.type[0] == 8:
                all_coordinates = annot.vertices
                #print(all_coordinates)
                if len(all_coordinates) == 4:
                    highlight_coord = fitz.Quad(all_coordinates).rect
                    highlights.append(highlight_coord)
                else:
                    all_coordinates = [all_coordinates[x:x+4] for x in range(0, len(all_coordinates), 4)]
                    for i in range(0,len(all_coordinates)):
                        coord = fitz.Quad(all_coordinates[i]).rect
                        highlights.append(coord)
            annot = annot.next

        all_words = page.get_text_words()
        temp = page.get_text()
        txt = txt + temp
        highlight_text = []
        if len(highlights) > 0:
            for h in highlights:
    #             sentence = [w[4] for w in all_words if  fitz.Rect(w[0:4]).intersect(h)]
                sentence = []
                for w in all_words:
                    if fitz.Rect(w[0:4]).intersects(h):
                        sentence.append(w[4])
                highlight_text.append(" ".join(sentence))
        if len(highlight_text) > 0:
            result.append(" ".join(highlight_text))
            
    hg_text = nltk.sent_tokenize("\n".join(result))

    return hg_text,txt

def write_to_file(rep,ranked,pre_list, all_pre, pre, empty_flag):

    rep = rep.split('.')[0] 
    rep = rep[rep.rfind("\\") + 1:] + '.txt'
    print(rep)
    path = os.path.join(r"C:\Users\ayush\Downloads",rep)

    if empty_flag:
        with open(path, "w") as f:
            f.write('Document has no highlighted text')
        
        return


    with open(path, "w") as f:
        f.write('Rank \t Sentence \t\t\t\t Score \t Relevance \n')
        char = ''
        for rank,sen in enumerate(list(ranked[0])):
            if rank in pre_list:
                char = '*'
            else:
                char = ''
            f.write(f'{rank + 1} \t {sen[0:5]} + "...." + {sen[-5:]} \t {all_pre[rank]: .5f} \t {char} \n')

    #resetting path
    path = os.path.join(r"C:\Users\ayush\Downloads",'Stats.txt')
    with open(path,'a') as f:
        f.write(f'{rep.split(".")[0]}\t{len(list(ranked[0]))}\t{len(pre_list)}\t{pre} \n')
    
    




def driver(rep_dir):
    #importing the BERT Model
    model,tokenizer = BERT_Model()
    kw_mean = kw_mean_gen(model,tokenizer)
    # directory = 'annoted_sustainability_reports'
    # headings = os.listdir(directory)
    # res = []

    #writing the header string to Stats.txt file
    # path = os.path.join(r"C:\Users\ayush\Downloads",'Stats.txt')
    # with open(path,'a') as f:
    #     f.write(f'Company \t Total # of sentences \t # of relevant sentences \t AP \n')

    

    # for head in headings[0:2]:
    #     new_dir = os.path.join(directory,head)
    #     reports = os.listdir(new_dir)
    #     empty_flag = 0
    #     for rep in reports[0:2]:

            # rep_dir = os.path.join(new_dir,rep)

            # print(rep_dir)
    empty_flag = 0
    #extract the text and the hg_text
    hg_text, txt = hg_txt_extractor(rep_dir)
    tokens = nltk.sent_tokenize(txt.replace('\n',' '))
            
    #if empty text document
    if not hg_text:
        empty_flag = 1

    #preprocessing the text
    pre_tokens = []
    for sen in tokens:
        if len(sen.split(' ')) > 3:
            pre_tokens.append(sen)

    #Generate Sentence Embeddings
    corpus = Sen_Embedder(pre_tokens,model,tokenizer)

    #Ranking Sentences
    ranked = ranking(corpus,kw_mean,pre_tokens)

    #calculate precision for each doc
    pre = 0
    pre_list = []
    all_pre = []
    if not empty_flag:
        pre, pre_list, all_pre = cal_pre(hg_text,ranked)

    #writing outputs to file 
    write_to_file(rep_dir, ranked, pre_list, all_pre, pre, empty_flag)

    # fin_map = sum(res)/len(res)

    # path = os.path.join(r"C:\Users\ayush\Downloads",'Stats.txt')

    # with open(path,'a') as f:
    #     f.write(f'{map} \n')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='A test program.')

    parser.add_argument("report_path", help="Calculator for AP")

    args = parser.parse_args()

    driver(args.report_path)



