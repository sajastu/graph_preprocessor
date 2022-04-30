
import argparse
import codecs
import json
import os
from multiprocessing import Pool

import numpy as np
import re
# Get a counter for the iterations
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import Counter
print("Loaded libraries...")
import nltk
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_lg')

STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
call can cannot ca could
did do does doing done down due during
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
few fifteen fifty first five for former formerly forty four from front full
further
get give go
had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred
i if in indeed into is it its itself
keep
last latter latterly least less
just
made make many may me meanwhile might mine more moreover most mostly move much
must my myself
name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
part per perhaps please put
quite
rather re really regarding
same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such
take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two
under until up unless upon us used using
various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
yet you your yours yourself yourselves
""".split()
)

stopwords_spacy = nlp.Defaults.stop_words

nltk.download('stopwords')
sws = stopwords.words('english') + ["get", "take", "n’t", '"', "ca", "nt", "'s", "got", "still", "...", ":", '"', "wo",
                                    "everything", "amp;#x200b", "_", "..", 'am', "'m", "'ve", "are", "is", "'s", ".", "n't", ",",
                                    "*", "feel", "much", "pretty", "also", "even", ")", "(", "want", "?", "/", "na", "'re", "-", "%"]

STOP_WORDS.update({"like", "going", "say", "took", "new", "tic", "lol"})

sws = list(set(sws + list(STOP_WORDS)))


parser = argparse.ArgumentParser(
    description="Builds an extractive summary from a json prediction.")

parser.add_argument('-dataset', type=str,
                    default='/disk1/sajad/datasets/medical/mental-reddit-reduced/sets/train-final.json',
                    help="""Path of the dataset file""")

parser.add_argument('-output', type=str,
					default='/disk1/sajad/datasets/medical/mental-reddit-reduced/sets/with-annotated/train-annotated-mental.json',
                    help="""Path of the output files""")


parser.add_argument('-prune', type=int, default=-1,
                   help="Prune to that number of words.")
parser.add_argument('-num_examples', type=int, default=100000,
                   help="Prune to that number of examples.")

opt = parser.parse_args()


def get_tokens(txt):
    out_tokens = []
    doc = nlp(txt)
    for sent in doc.sents:
        for word in sent:
            # out_tokens.append((word.text, word.lemma_))
            out_tokens.append(word.text)
    return out_tokens


def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end+1])

def format_json(s):
    return json.dumps({'sentence':s})+"\n"

def splits(s, num=2000):
    # tokenize the source
    if num != -1:
        # return s.split()[:num]
        return get_tokens(s)[:num]
    else:
        # return s.split()
        return get_tokens(s)


def check_mental_ontology(counter, t, substring):
    # if substring == 'panic attack':
    #     print('2')
    #     import pdb;
    #     pdb.set_trace()
    # print(f'c({counter}) -s ({substring})')
    matches_from_ont = set()
    for j, tt in enumerate(t):
        minor_counter = counter

        substring_found = True
        if len(tt.split()) > minor_counter:
            while minor_counter > -1:
                try:
                    if substring.split()[minor_counter].lower() == tt.split()[minor_counter].lower():
                        matches_from_ont.add(tt.lower())
                        # if substring == 'panic attack':
                        #     import pdb;pdb.set_trace()
                        minor_counter -= 1
                        continue
                    else:
                        substring_found = False
                        break
                except:
                    continue
        else:
            substring_found = False

        if substring_found:
            return True, substring in list(matches_from_ont)
        else:
            continue

    return False, substring in list(matches_from_ont)

def make_BIO_tgt_mental(s, t):
    # t should be an array or str
    # tsplit = t.split()
    ssplit = s#.split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    loop_counter = 0
    t = [tt for tt in t if len(tt.strip()) > 0]
    ssplit = [s for s in ssplit if len(s.strip()) > 0]
    prev_partial_match = False
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)


        assert loop_counter == len(searchstring.split()) - 1, "Loop counter shouldn't be 1 less than search query's length"

        partial_match, exact_match = check_mental_ontology(loop_counter, t, searchstring)

        if exact_match:
            loop_counter = 0
            full_string = compile_substring(startix, endix, ssplit)
            matches.extend(["1"] * (endix - startix + 1))
            matchstrings[full_string] += 1
            endix += 1
            startix = endix

        elif (partial_match) and endix < len(ssplit)-1:
        # if searchstring in t and endix < len(ssplit)-1 and searchstring == 'panic attack':
            endix += 1
            loop_counter += 1
            prev_partial_match = True

        else:
            # if searchstring == 'a panic':
            #     import pdb;pdb.set_trace()

            loop_counter = 0
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            # if startix >= endix:#-1:
            if not prev_partial_match:
                matches.extend(["0"] * (endix - startix + 1))
                endix += 1
                prev_partial_match = False
            else:
                matches.extend(["0"] * (endix - startix))
                endix = endix
                prev_partial_match = False


            startix = endix

    return matches

def make_BIO_tgt(s, t, mental_retrieve=False):
    # t should be an array or str
    # tsplit = t.split()
    ssplit = s#.split()

    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)

        if searchstring in t and endix < len(ssplit)-1:
            endix += 1

        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:#-1:
                matches.extend(["0"] * (endix-startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix-1, ssplit)
                matches.extend(["1"]*(endix-startix))
                matchstrings[full_string] +=1

            startix = endix

    return matches


def retrieve_mental_string():
    out_str = []
    with open('mental_illnesses.txt') as fR:
        for l in fR:
            out_str += [l.strip()]

    return out_str

def _mp_labelizer(instance):
    idd, ssplits, t, mental_ds = instance
    # tgt = make_BIO_tgt(ssplits, t)
    tgt_mental = make_BIO_tgt_mental(ssplits, mental_ds)
    return idd, ssplits, None, tgt_mental

def main():
    lcounter = 0
    max_total = opt.num_examples

    DS_PATH = opt.dataset
    SOURCE_PATH = 'src.txt'
    TARGET_PATH = 'tgt.txt'
    ID_PATH = 'id.txt'

    NEW_TARGET_PATH = opt.output + ".txt"
    PRED_SRC_PATH = opt.output + ".pred.txt"
    PRED_TGT_PATH = opt.output + ".src.txt"

    # SOURCE_PATH and TARGET_PATH should be created here and will be removed at the end
    json_ents = {}
    idd = 0
    with codecs.open(SOURCE_PATH, 'w', "utf-8") as fWS, codecs.open(TARGET_PATH, 'w', "utf-8") as fWT, codecs.open(ID_PATH, 'w', 'utf-8') as fWI:
        with open(DS_PATH) as fR:
            for l in fR:
                ent = json.loads(l.strip())
                src = ent['src']
                tldr = ent['tldr']
                id = idd
                json_ents[str(id)] = ent
                fWS.write(src)
                fWS.write('\n')
                fWT.write(tldr)
                fWT.write('\n')
                fWI.write(str(id))
                fWI.write('\n')
                idd+=1


    with codecs.open(SOURCE_PATH, 'r', "utf-8") as sfile:
        for ix, l in enumerate(sfile):
            lcounter += 1
            if lcounter >= max_total:
                break

    sfile = codecs.open(SOURCE_PATH, 'r', "utf-8")
    tfile = codecs.open(TARGET_PATH, 'r', "utf-8")
    ifile = codecs.open(ID_PATH, 'r', "utf-8")
    outf = codecs.open(NEW_TARGET_PATH, 'w', "utf-8", buffering=1)
    outf_tgt_src = codecs.open(PRED_SRC_PATH, 'w', "utf-8", buffering=1)
    outf_tgt_tgt = codecs.open(PRED_TGT_PATH, 'w', "utf-8", buffering=1)


    actual_lines = 0
    mental_ilnesses_string = retrieve_mental_string()
    splits_to_be_processed = []
    for ix, (s, t, id) in tqdm(enumerate(zip(sfile, tfile, ifile)), total=lcounter):
        ssplit = splits(s, num=opt.prune)
        # Skip empty lines

        if len(ssplit) < 2 or len(t.split()) < 2:
            continue
        else:
            actual_lines += 1
        # Build the target

        splits_to_be_processed.append((id, ssplit, t, mental_ilnesses_string))

        # ssplit = ['I', 'got', 'a', 'panic', 'attack', 'recently', '.', "Also", "I", "have", "adhd"]
        # import pdb;pdb.set_trace()

    pool = Pool(2)
    for labels in tqdm(pool.imap_unordered(_mp_labelizer, splits_to_be_processed), total=len(splits_to_be_processed)):
        id, ssplit, tgt, tgt_mental = labels

        # tgt = [int(t) + int(tm) for t, tm in zip(tgt, tgt_mental)]
        # tgt_par = [(s,t) for s,t in zip(ssplit, tgt_mental.split())]
        # Format for allennlp
        tgt = tgt_mental

        annotations = []
        seen_tokens_len = 0
        sstring = ' '.join(ssplit)
        for j, (token, tag) in enumerate(zip(ssplit, tgt)):
            if (tag == 1 or tag==2) and token.lower() not in sws:
                annotations.append(
                    {
                        'start': seen_tokens_len,
                        'end': seen_tokens_len + len(token),
                        'text': token,
                        'label': tag,
                    }
                )

            seen_tokens_len += len(token)+1

        # import pdb;pdb.set_trace()
        json_ents[id.strip()]['src'] = sstring.strip()
        json_ents[id.strip()]['annotations'] = {}
        json_ents[id.strip()]['annotations'] = annotations
        # for token, tag in zip(ssplit, tgt.split()):
        #     outf.write(token+"###"+tag + " ")

    sfile.close()
    tfile.close()
    outf.close()
    outf_tgt_src.close()
    outf_tgt_tgt.close()
    os.remove(SOURCE_PATH)
    os.remove(TARGET_PATH)

    # writing new and updated json ents...
    with open(opt.output, mode='w') as fW:
        for key, ins in json_ents.items():
            json.dump(ins, fW)
            fW.write('\n')


if __name__ == "__main__":
    main()