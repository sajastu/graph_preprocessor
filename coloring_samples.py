import json

import spacy
import pandas as pd
nlp = spacy.load("en_core_web_md")
# nlp.max_length = 1500000
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import to_hex

import xlsxwriter
from tqdm import tqdm
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

requested_doc_id = 'test-292'


# read examples...

def get_sentences(text):
    doc = nlp(text)
    out_sents = []
    for sent in doc.sents:
        if len(sent) > 1:
            out_sents.append(sent.text)
    return out_sents

def get_rouge(pred_sentences, ref, type='rouge2'):
    out_scores = []
    for sent in pred_sentences:
        scores = scorer.score(ref, sent)
        out_scores.append(scores[type].fmeasure)
    return out_scores

def get_rouge_with_src(src_sents, sentences_1, k):

    context = []

    for sum_sent in sentences_1:
        rg_scores = []
        for j, src_sent in enumerate(src_sents):
            rg_scores.append((j, src_sent, scorer.score(sum_sent, src_sent)['rougeL'].fmeasure))
        top_sents = sorted(rg_scores, key=lambda x:x[2], reverse=True)[:k]
        top_sents = sorted(top_sents, key=lambda x:x[0])
        context.append(' '.join([ f'----({sent[0]+1})({sent[2]})----- {sent[1]} \n' for z, sent in enumerate(top_sents)]))

    return context

def get_format(workbook, colors):
    formats = []

    for color in colors:
        formats.append(workbook.add_format({'bg_color': color}))

    return formats


def retrieve_src(doc_id):
    with open('/disk1/sajad/gov-reports/gov-reports/test-withIds.json') as fR:
        for l in fR:
            ent = json.loads(l)
            if ent['doc_id'] == doc_id:
                return ent['source']

def main():
    results_dict = pd.read_csv('led_grease1.csv').to_dict()



    cmap = cm.get_cmap('Greens')

    baseline_summaries = results_dict['led-generated']
    model_summaries = results_dict['grease1-generated']
    gold_summaries = results_dict['summary']
    src_text = retrieve_src(requested_doc_id)
    doc_ids = results_dict['doc_id']

    req_idx = [j for j, id in enumerate(list(doc_ids.values())) if id == requested_doc_id][0]
    i=0
    # import pdb;pdb.set_trace()

    while i < len(list(gold_summaries)[req_idx:req_idx+1]):

        workbook = xlsxwriter.Workbook(f'analysis/{doc_ids[req_idx]}.xlsx')
        worksheet_sum1 = workbook.add_worksheet()

        gold = gold_summaries[req_idx]
        sum1 = baseline_summaries[req_idx]
        sum2 = model_summaries[req_idx]

        sentences_1 = get_sentences(sum1)
        sentences_2 = get_sentences(sum2)
        gold_sents = get_sentences(gold)
        src_sents = get_sentences(src_text)

        # calculate_rg
        rg_sum1 = get_rouge(sentences_1, gold, type='rougeL')
        rg_sum2 = get_rouge(sentences_2, gold, type='rougeL')

        sum_1_src_sents = get_rouge_with_src(src_sents, sentences_1, k=5)
        sum_2_src_sents = get_rouge_with_src(src_sents, sentences_2, k=5)
        gold_src_sents = get_rouge_with_src(src_sents, gold_sents, k=5)


        norm = Normalize(vmin=min(rg_sum1), vmax=max(rg_sum1))
        rgba_values_1 = cmap(norm(rg_sum1))

        hex_1 = [to_hex([rgba_values_1[i][0], rgba_values_1[i][1], rgba_values_1[i][2], rgba_values_1[i][3]]) for i in range(rgba_values_1.shape[0]) ]

        norm = Normalize(vmin=min(rg_sum2), vmax=max(rg_sum2))

        rgba_values_2 = cmap(norm(rg_sum2))
        hex_2 = [to_hex([rgba_values_2[i][0], rgba_values_2[i][1], rgba_values_2[i][2], rgba_values_2[i][3]]) for i in range(rgba_values_2.shape[0]) ]

        formats_1 = get_format(workbook, hex_1)
        formats_2 = get_format(workbook, hex_2)

        print('baseline sentences')
        for row, sent in enumerate(sentences_1):
            worksheet_sum1.write(row, 2, sum_1_src_sents[row])
            worksheet_sum1.write_number(row, 1, rg_sum1[row], formats_1[row])
            worksheet_sum1.write(row, 0, sent)

        worksheet_sum1.write(row + 1, 0, 'Grease', workbook.add_format({'bg_color': 'black', 'font_color': 'white', 'bold': True}))
        worksheet_sum1.write(row + 1, 1, '', workbook.add_format({'bg_color': 'black'}))
        worksheet_sum1.write(row + 1, 2, '', workbook.add_format({'bg_color': 'black'}))
        worksheet_sum1.write(row + 1, 3, '', workbook.add_format({'bg_color': 'black'}))
        print('model sentences')

        for row, sent in enumerate(sentences_2):
            worksheet_sum1.write(row + len(sentences_1) + 1, 2, sum_2_src_sents[row])
            worksheet_sum1.write_number(row + len(sentences_1) + 1, 1, rg_sum2[row], formats_2[row])
            worksheet_sum1.write(row + len(sentences_1) + 1, 0, sent)


        worksheet_sum1.write(row + len(sentences_1) + 2, 0, 'Summary', workbook.add_format({'bg_color': 'black', 'font_color': 'white', 'bold': True}))
        worksheet_sum1.write(row + len(sentences_1) + 2, 1, '', workbook.add_format({'bg_color': 'black'}))
        worksheet_sum1.write(row + len(sentences_1) + 2, 2, '', workbook.add_format({'bg_color': 'black'}))
        worksheet_sum1.write(row + len(sentences_1) + 2, 3, '', workbook.add_format({'bg_color': 'black'}))

        print('summary sentences')

        for row, sent in enumerate(gold_sents):
            worksheet_sum1.write(row + len(sentences_1) + len(sentences_2) + 2, 2, gold_src_sents[row])
            worksheet_sum1.write(row + len(sentences_1) + len(sentences_2) + 2, 1, '---')
            worksheet_sum1.write(row + len(sentences_1) + len(sentences_2) + 2, 0, sent)

        workbook.close()
        print('done')

        i+=1

        # to hex

        # pack it together with sentences






if __name__ == '__main__':
    main()