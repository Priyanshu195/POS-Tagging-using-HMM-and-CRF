import numpy as np
import string
import nltk
from collections import defaultdict
from nltk.corpus import brown
import time
from sklearn_crfsuite import CRF
import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_sent(sent):
    # Convert to lowercase and split into words
    return [word for word in sent.split()]

def Viterbi_decoder(t_mat, e_mat, sent):
    words = preprocess_sent(sent)
    n = len(words)
    prob_mat = np.zeros((n, n_tags))
    tag_mat = np.zeros((n, n_tags), dtype=int)

    # Initialize the first column of the probability matrix
    first_word_index = token_to_index.get(words[0], -1)

    # Correctly initialize probabilities for the first word
    for i in range(n_tags):
        prob_mat[0][i] = t_mat[tag_to_index["START"]][i] * (e_mat[i][first_word_index] if first_word_index != -1 else 1e-10)

    # Fill the probability and tag matrices
    for s in range(1, n):
        curr_token = words[s]
        curr_token_index = token_to_index.get(curr_token, -1)

        for t in range(n_tags):
            max_prob, prev_tag_index = max(
                (prob_mat[s-1][u] * t_mat[u][t], u) for u in range(n_tags)
            )
            prob_mat[s][t] = max_prob * (e_mat[t][curr_token_index] if curr_token_index != -1 else 1e-10)
            tag_mat[s][t] = prev_tag_index

    # Backtrack to find the best path
    pred = []
    last_tag = np.argmax(prob_mat[-1])
    pred.append(tags[last_tag])

    for i in range(n - 1, 0, -1):
        last_tag = tag_mat[i][last_tag]
        pred.append(tags[last_tag])

    pred.reverse()
    return words, pred

def sent2features(sent):
    features = []
    for i, word in enumerate(sent):
        feature = {
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            feature['prev_word.lower()'] = sent[i - 1].lower()
        else:
            feature['BOS'] = True  # Beginning of sentence
        if i < len(sent) - 1:
            feature['next_word.lower()'] = sent[i + 1].lower()
        else:
            feature['EOS'] = True  # End of sentence
        features.append(feature)
    return features

# Build the tokens and tags from the training set
# brown_words = [word for sent in train_sents for word, tag in sent]
# brown_tags = [tag for sent in train_sents for word, tag in sent]
# tokens = list(set(brown_words))
tags = ['START', 'PRT', 'NOUN', 'PRON', 'VERB', 'CONJ', 'X', 'ADJ', 'ADP', 'DET', 'NUM', '.', 'ADV']
n_tags = len(tags)


# Get the HMM model
with open('hmm_model.pkl', 'rb') as f:
    model = pickle.load(f)
    tm = model['transition_matrix']
    em = model['emission_matrix']
    token_to_index = model['token_to_index']
    tag_to_index = model['tag_to_index']

# Get the CRF model
with open('crf_model.pkl', 'rb') as f:
    crf_model = pickle.load(f)

# Streamlit application for POS tagging
st.title("POS Tagging with HMM and CRF")
sentence = st.text_input("Enter a sentence:")

if st.button("Tag the Sentence"):
    if sentence:
        # Timing and prediction with HMM
        hmm_start = time.perf_counter()  # Use perf_counter for better precision
        output_hmm = Viterbi_decoder(tm, em, sentence)
        hmm_end = time.perf_counter()

        # Timing for CRF
        crf_start = time.perf_counter()
        sent_features = sent2features(preprocess_sent(sentence))
        crf_output = crf_model.predict([sent_features])[0]  # Using the trained CRF model
        crf_end = time.perf_counter()

        result_df = pd.DataFrame({'words': output_hmm[0],'hmm': output_hmm[1], 'crf': crf_output})

        st.subheader("POS Tagging Results")
        st.dataframe(result_df)

        st.write(f"HMM Prediction Time: {hmm_end - hmm_start:.4f} seconds")
        st.write(f"CRF Prediction Time: {crf_end - crf_start:.4f} seconds")
    else:
        st.warning("Please enter a sentence!")

