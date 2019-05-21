import argparse
import os
import sys
from gensim.models import Word2Vec as W2V
import numpy as np
import csv

###################################
# Definitions
###################################

model_fn = 'cases_w2v.bin'
output_fn = 'outputData.csv'

argparser = argparse.ArgumentParser(description='A question answering system')
argparser.add_argument('path',help='Specify the path to the input file')
argparser.add_argument('--model', help='Specify which model to use, A or B', choices=['A', 'B'])
args = argparser.parse_args()

if args.model is None:
    args.model = 'A'


###################################
# Check environment
###################################

if not args.path:
    print "Error: Please specify an input filename."
    sys.exit()

no_path = (not os.path.exists(model_fn))
no_model = (not os.path.exists(args.path))
# model_spec = (args.model in ['A','B'])
error_s = ""
if no_path:
    error_s += "Error: Could not find embeddings file %s\n" % model_fn
if no_model:
    error_s += "Error: Could not find input file %s\n" % args.path
#if not model_spec:
#    error_s += "Error: Choose either model 'A' or 'B'. Specified: '%s'\n" % args.model
if no_path or no_model: # or (not model_spec):
    print error
    sys.exit()


###################################
# Load the Word2Vec model
###################################

print "Loading the Word2Vec model ..."
w2v = W2V.load_word2vec_format('cases_w2v.bin', binary=True)

def similarity(ta, tb):  # Return similarity between tokens if they exist
    if (ta in w2v.vocab) and (tb in w2v.vocab):
        return w2v.similarity(ta, tb)
    else:
        return 0


###################################
# Preprocessing functions
###################################

from nltk import download
download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer

porter = PorterStemmer()
stopw = set(stopwords.words('english'))

def clean_token(w):  # Cleans a single token
    w = porter.stem(''.join([a.lower() for a in w if a.isalnum()]))
    return w

def clean_phrase(s):  # Cleans a phrase string
    s = s.split()
    s = ' '.join([clean_token(w) for w in s])
    return s

def remove_stopw(s):  # Removes stopwords and cleans a phrase
    s = clean_phrase(s)
    s = ' '.join([i for i in wordpunct_tokenize(s) if i not in stopw])
    return s


###################################
# Load data
###################################

print "Loading the data in %s ..." % args.path
with open(args.path, 'r') as infile:
    reader = csv.reader(infile)
    data = [row for row in reader]

questions_original = [d[0] for d in data]
answers_original = [d[1:] for d in data]

print "Preprocessing the data ..."
questions = [remove_stopw(d[0]) for d in data]
answers = [map(remove_stopw, d[1:]) for d in data]


###################################
# Model processing
###################################

# Scores for a given model
def model_scores(model, qapair=None):
    if qapair is None:  # Do all
        scores = [[]]*len(questions)
        for i in xrange(len(questions)):
            scores[i] = model(questions[i], answers[i])
    else:
        scores = model(questions[qapair], answers[qapair])
    return scores

# Model A - Log Word2Vec similarities
def pairwise_similarity_log(question, candidates, get_sim=False):
    qtokens = list(set(question.split()))
    scores = [0]*len(candidates)
    for i, c in enumerate(candidates):
        candidate = c.split()
        words = {}
        [words.update([(qt+ct, similarity(qt, ct))]) for qt in qtokens for ct in candidate]
        similarities = words.values()
        scores_pos = [np.log(k+1) for k in similarities if k > 0]
        scores_neg = [-np.log(-k+1) for k in similarities if k < 0]
        similarities = scores_neg+scores_pos
        scores[i] = (sum(scores_pos)-sum(scores_neg))/((len(qtokens)*len(candidate)) - len([s for s in similarities if s == 0]))
    if get_sim:
        return similarities
    else:
        return scores

# Model B - Question words fraction
def similarity_log_completion(question, candidates, get_sim=False):
    qtokens = list(set(question.split()))
    scores = [0] * len(candidates)
    for i, c in enumerate(candidates):
        candidate = c.split()
        words = {}
        [words.update([(qt, words.setdefault(qt, 0) + similarity(qt, ct))]) for qt in qtokens for ct in candidate]
        similarities = words.values()
        scores_pos = [np.log(k + 1) for k in similarities if k > 0]
        scores_neg = [-np.log(-k + 1) for k in similarities if k < 0]
        similarities = scores_neg + scores_pos
        scores[i] = (sum(scores_pos) - sum(scores_neg)) / (
        (len(qtokens) * len(candidate)) - len([s for s in similarities if s == 0]))
    if get_sim:
        return similarities
    else:
        return scores

models = {
    'A': pairwise_similarity_log,
    'B': similarity_log_completion
}
model = models[args.model]

###################################
# Answer selection
###################################

def answer_selection(model):
    scores = model_scores(model)
    guesses = map(lambda x: x.index(max(x)), scores)
    guess_answers = [answers_original[i][guesses[i]] for i, score in enumerate(scores)]
    return zip(questions_original, guess_answers)

def save_output(output_data):
    with open(output_fn, 'w') as outfile:
        writer = csv.writer(outfile)
        for row in output_data:
            writer.writerow(row)
    print "Done! Results saved to %s" % output_fn

print "Answering questions ... Please wait."
output_data = answer_selection(model)
save_output(output_data)