import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ast
 
 
BOT_CONFIG = {}

with open('config.txt', 'r') as f:
    content = f.read()
    BOT_CONFIG = ast.literal_eval(content)
 
# DATASET PREPARATION
dataset = []
for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        dataset.append([example, intent])

# VECTORIZATION
corpus = [text for text, intent in dataset]
y = [intent for text, intent in dataset]

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3)) # also can use vectorizers  frm sklearn like LinearSVC() 
X = vectorizer.fit_transform(corpus)

# CLASSIFICATION
clf = SVC(probability=True) # also can use different algorithms from sklearn like TfidfVectorizer 
clf.fit(X, y) # train the model

# VALIDATION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # using bilt_in sklearn function to mix the data 
clf.fit(X_train, y_train) # train the model
# print(clf.score(X_test, y_test)) # accuracy


defaultDataSet = []
questions = set()
search_dataset = {}


def clear_text(text):
    text = text.lower()
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz0123456789- '
    res = ''
    for c in text:
        if c in alphabet:
            res += c
    return res

def get_intent(text):
    proba_list = clf.predict_proba(vectorizer.transform([text]))[0]
    max_proba = max(proba_list)
    if max_proba > 0.1:
        index = list(proba_list.index(max))
        return clf.classes[index]

def get_response_by_intent(intent):
    phrases = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(phrases)

# Preparaton of dowloaded dataset with answers and questions

with open('dialogues.txt') as f:
    content = f.read()

blocks = content.split('\n\n')
for block in blocks:
    replicas = block.split('\n')[:2]
    if len(replicas) == 2:
        question = replicas[0][2:]
        answer = replicas[1][2:]
        if answer and question and question not in questions:
            questions.add(question)
            defaultDataSet.append([question, answer])

for question, answer in defaultDataSet:
    words = question.split(' ')
    for word in words:
        if word not in search_dataset:
            search_dataset[word] = []
        search_dataset[word].append((question, answer))

search_dataset = {
    word: word_dataset
    for word, word_dataset in search_dataset.items()
    if len(word_dataset) < 1000
}


def get_response_generatively(text):
    text = clear_text(text)
    if not text:
        return
    words = text.split(' ')

    words_dataset = set()
    for word in words:
        if word in search_dataset:
            words_dataset |= set(search_dataset[word])

    scores = []

    for question, answer in words_dataset:
        if abs(len(text) - len(question)) / len(question) < 0.4:
            distance = nltk.edit_distance(text, question)  # cheking similarity of the words
            score = distance / len(question)
            if score < 0.3:
                scores.append([score, question, answer])
    if scores:
        return min(scores, key=lambda s: s[0])[2]

def get_failure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


def bot(request):
    # NLU
    intent = get_intent(request)

    # response generation
    if intent:
        return get_response_by_intent(intent)
    
    response = get_response_generatively(request)
    if response:
        return response
    
    return get_failure_phrase()

print(bot('привет'))