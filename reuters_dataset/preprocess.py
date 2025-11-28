import spacy
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
import torch

# Load spaCy's English model (disable parser and NER for speed)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Load and lowercase NLTK stopwords
stop_words = set(word.lower() for word in stopwords.words('english'))

# Optional: only keep certain parts of speech (POS)
# Common choices: NOUN (common noun), PROPN (proper noun), ADJ (adjective)
allowed_postags = {"NOUN", "PROPN", "ADJ", "ADV", "VERB"}

def preprocess(doc,
               allowed_postags=allowed_postags,
               stop_words=stop_words):
    """
    Preprocesses a single document:
    - Tokenizes and lemmatizes using spaCy
    - Keeps only alphabetic tokens
    - Filters stopwords
    - Filters short tokens
    - Keeps only desired POS tags
    - Lowercases final tokens

    Input: a single document, a string
    Output: a list of tokens
    """
    doc = nlp(doc)                                  # split the string into list of tokens
    return [
        token.lemma_.lower() for token in doc       # lemmatization, lowercase
        if token.pos_ in allowed_postags            # choose speific kinds of words, like noun
        and token.is_alpha                          # remove numbers, punctuations, etc.
        and token.lemma_.lower() not in stop_words  # remove stopwords defined in nltk
        and len(token) > 2                          # remove the words of less than 3 letters
    ]

def dictionarize(tokenized_docs, 
                 no_below=5, 
                 no_above=0.5):
    """
    Given a corpus of tokenized documents, 
    which is a list of lists of tokens,
    filter the tokens that appear in less than no_below 
    documents, or that appear in more than no_above of all
    documents

    Input:
    - tokenized documents, a list of lists of tokens
    - no_below
    - no_above

    Return:
    - filtered_tokenized_docs
    - bag-of-word of filtered_tokenized_docs
    - dictionary, as defined in Gensim
    """
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    allowed_tokens = set(dictionary.token2id)
    filtered_tokenized_docs = [[token for token in doc if token in allowed_tokens] 
                               for doc in tokenized_docs]
    
    bow = [dictionary.doc2bow(doc) for doc in filtered_tokenized_docs]

    return filtered_tokenized_docs, \
           bow, \
           dictionary

def bow2coo(bow, dictionary):
    """
    Given the bag-of-word representation of a corpus,
    and the corresponding dictionary,
    return the pytorch coo sparse matrix
    """
    # Step 1: Build index lists and values
    rows = []
    cols = []
    values = []

    for doc_id, doc in enumerate(bow):
        for token_id, count in doc:
            rows.append(doc_id)
            cols.append(token_id)
            values.append(count)

    # Step 2: Convert to tensors
    indices = torch.tensor([rows, cols], dtype=torch.long)  # shape: (2, nnz)
    values = torch.tensor(values, dtype=torch.float64)      # shape: (nnz,), notice datatype
    shape = (len(bow), len(dictionary))                     # shape: (D, V)

    # Step 3: Create sparse tensor
    sparse_matrix = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    return sparse_matrix
