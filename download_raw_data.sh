# download ConceptNet
mkdir -p /dev/disk/data/
mkdir -p /dev/disk/data/cpnet/
wget -nc -P /dev/disk/data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd /dev/disk/data/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
# download ConceptNet entity embedding
wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../




# download CommensenseQA dataset
#mkdir -p /dev/disk/data/csqa/
#wget -nc -P /dev/disk/data/csqa/ https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
#wget -nc -P /dev/disk/data/csqa/ https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
#wget -nc -P /dev/disk/data/csqa/ https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
#
## create output folders
mkdir -p /dev/disk/data/cnn-dm/ground/
mkdir -p /dev/disk/data/cnn-dm/graph/
mkdir -p /dev/disk/data/cnn-dm/statement/
python news_datasets/generate_text_from_datasets.py



# download OpenBookQA dataset
#wget -nc -P data/obqa/ https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip
#yes n | unzip data/obqa/OpenBookQA-V1-Sep2018.zip -d data/obqa/

# create output folders
#mkdir -p data/obqa/fairseq/official/
#mkdir -p data/obqa/grounded/
#mkdir -p data/obqa/graph/
#mkdir -p data/obqa/statement/
