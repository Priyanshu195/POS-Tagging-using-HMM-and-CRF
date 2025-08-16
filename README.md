# POS-Tagging-using-HMM-and-CRF
This project is part of the course CS626: Speech, Natural Language Processing, and the Web, instructed by Prof. Pushpak Bhattacharyya. It focuses on building and evaluating Part of Speech (POS) tagging models using two prominent approaches: Hidden Markov Model (HMM) and Conditional Random Field (CRF). The dataset used is the Brown Corpus with the universal tagset.

Project Components
Model Files (.pkl): Pre-trained HMM and CRF models are stored in the provided .pkl files.
GUI Interface (gui_hmm_crf.py): A user-friendly interface built using Streamlit to allow users to input text and visualize POS tagging using either the HMM or CRF models.
Model Training Notebook (hmm_crf_model_training.ipynb): Contains the code used for training both the HMM and CRF models on the Brown Corpus.
Key Features
POS Tagging Models: We developed two distinct models for POS tagging:

HMM-based model: A probabilistic generative model that assigns tags based on observed sequences and hidden states.
CRF-based model: A discriminative model that considers the relationships between adjacent tags and features of the input sequence.
Performance Comparison:

We rigorously evaluated the performance of both models using key metrics such as:
Accuracy
Precision (P)
Recall (R)
F-score
Confusion matrix
Per POS tag accuracy
These metrics were used to highlight strengths and weaknesses of each model, showing scenarios where one outperforms the other.
Demonstrations:

We present examples showcasing:
Situations where the HMM model performs better.
Cases where the CRF model outshines the HMM.
Instances where both models yield comparable results.
ChatGPT Comparison: For additional context, we briefly compare the performance of the HMM model against a baseline provided by ChatGPT on POS tagging.

Conclusion
This project illustrates the strengths and trade-offs between the HMM and CRF approaches to POS tagging. The GUI tool provides an intuitive way for users to experiment with both models and observe their performance in real-time.

