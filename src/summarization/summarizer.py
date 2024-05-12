import pymorphy3
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel

import re
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
    
stop_words = set(stopwords.words("russian"))

def preprocess_text(text):

    
    morph = pymorphy3.MorphAnalyzer()

    sentences = []
    for sentence in text:
        sentence = re.sub(r"[^\w\s]", "", sentence)
        lemmatized = []
        for word in sentence.split(' '):
            if word.lower() not in stop_words:
                lemmatized.append(morph.parse(word.lower())[0].normal_form)
        sentences.append(' '.join(lemmatized))
    
    return sentences

def get_embedding(text):
    tokenized_texts = [tokenizer(i, padding=True, truncation=True, return_tensors="pt") for i in text]
    with torch.no_grad():
        outputs = [model(**i).last_hidden_state[:, 0, :] for i in tokenized_texts]
    return outputs

def summarize(text):
    text = text.split('.')

    proccessed_text = preprocess_text(text)

    embeddings = get_embedding(proccessed_text)

    similarities = calculate_similarity(embeddings)
    rank = ranging(similarities, len(proccessed_text))
    result = sorted(rank, key=lambda x: x[0], reverse=True)
    print(result)
    percent = int(len(result) * 0.3)
    result = result[:percent]
    return '. '.join([text[i[1]] for i in result]) + '.'


def ranging(similarities, ln):
    result = [[0, i] for i in range(ln)]
    for i in similarities:
        print(i[0])
        if i[0] > 0.95:
            result[i[1]][0] += 1
            result[i[2]][0] += 1
    return result
    
def calculate_similarity(sentence_embedding):
    similarity_matrix = []

    for i in range(len(sentence_embedding)):
        for j in range(i+1, len(sentence_embedding)):
            embedding1 = sentence_embedding[i]
            embedding2 = sentence_embedding[j]
            if isinstance(embedding1, int) or isinstance(embedding2, int):
                continue

            if embedding1.all() == 0 or embedding2.all() == 0:
                continue

            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            similarity_matrix.append((similarity, i, j))

    return similarity_matrix