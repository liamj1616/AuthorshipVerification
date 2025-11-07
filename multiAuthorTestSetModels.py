import os, re, random
import numpy as np

def extractInfoFromFileName(fileName):
    id = ''
    gender = ''
    age = ''
    topic = ''
    zodiacSign = ''

    readID = False
    readGender = False
    readAge = False
    readTopic = False
    readZodiacSign = False

    for char in fileName:
        if readID:
            if char == '.':
                readID = False
                readGender = True
                continue

            else:
                id += char
                continue

        elif readGender:
            if char == '.':
                readGender = False
                readAge = True
                continue

            else:
                gender += char
                continue

        elif readAge:
            if char == '.':
                readAge = False
                readTopic = True
                continue

            else:
                age += char
                continue

        elif readTopic:
            if char == '.':
                readTopic = False
                readZodiacSign = True
                continue

            else:
                topic += char
                continue

        elif readZodiacSign:
            if char == '.':
                break

            else:
                zodiacSign += char
                continue

        elif char == "/":
            readID = True

    return [id, gender, int(age), topic, zodiacSign]

def readFileData(fileName, label):
    texts = []
    labels = []
    openTag = '<post>'
    closeTag = '</post>'
    isTag = ''
    isReading = False

    fileContents = open(fileName, 'r', errors='ignore')

    while True:
        char = fileContents.read(1)
        if not char:
            break
        if isReading:
            if isTag != '':
                isTag += char
                if len(isTag) == len(closeTag):
                    if isTag == closeTag:
                        isReading = False
                    else:
                        texts[-1] += isTag
                    isTag = ''
                    continue
                else:
                    if isTag != closeTag[:len(isTag)]:
                        texts[-1] += isTag
                        isTag = ''
                    continue
            elif char == closeTag[0]:
                isTag = char
            else:
                texts[-1] += char

        else:
            if isTag != '':
                isTag += char
                if len(isTag) == len(openTag):
                    if isTag == openTag:
                        isReading = True
                        texts.append('')
                        labels.append(label)
                    isTag = ''
                    continue
                else:
                    if isTag != openTag[:len(isTag)]:
                        isTag = ''
                    continue
            elif char == openTag[0]:
                isTag = char

    assert(len(texts)) == len(labels)

    return texts, labels

def buildTestSet(test_set, test_labels, testSetAuthorMetadata, authorInQuestionFileName):
    for file in os.listdir('blogs'):
        if random.random() < 0.025:
            fileName = 'blogs/' + file
            if fileName != authorInQuestionFileName:
                filePosts, fileLabels = readFileData(fileName, 0)
                metadata = extractInfoFromFileName(fileName)
                for i in range(len(filePosts)):
                    if random.random() < (1/len(filePosts)):
                        test_set.append(filePosts[i])
                        test_labels.append(fileLabels[i])
                        testSetAuthorMetadata.append(metadata)

    return test_set, test_labels, testSetAuthorMetadata

# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

import transformers
import torch
import tensorflow as tf

from transformers import BertTokenizer

def preprocess_training_data(tokenizer, data):
    tokenized_data = tokenizer.batch_encode_plus(
        data,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )

    return {
        'input_ids': tf.convert_to_tensor(tokenized_data['input_ids']),
        'attention_mask': tf.convert_to_tensor(tokenized_data['attention_mask'])
    }

def preprocess_test_data(tokenizer, data):
    tokenized_data = tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'        
    )

    return {
        'input_ids': tf.convert_to_tensor(tokenized_data['input_ids']),
        'attention_mask': tf.convert_to_tensor(tokenized_data['attention_mask'])
    }

def output_results(results):
    confusionMatrix = [0, 0, 0, 0]
    '''
    - confusionMatrix[0]: true anomalies
    - confusionMatrix[1]: true normals
    - confusionMatrix[2]: false anomalies
    - confusionMatrix[3]: false normals
    '''
    for i in range(len(results)):
        if results[i] == 1 and test_labels[i] == 1:
            confusionMatrix[1] += 1
        elif results[i] == 1 and test_labels[i] != 1:
            confusionMatrix[3] += 1
        elif results[i] != 1 and test_labels[i] == 1:
            confusionMatrix[2] += 1
        elif results[i] != 1 and test_labels[i] != 1:
            confusionMatrix[0] += 1

    print('[true anomalies, true normals, false anomalies, false normals]')
    print(confusionMatrix)
    print('Accuracy: how many correct selections did the model make')
    accuracy = (confusionMatrix[0] + confusionMatrix[1]) / len(results)
    print('Precision: out of all selected to be anomalies, how many were actually anomalies')
    precision = confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[2])
    print('Recall: out of all anomalies, how many were selected as anomalies')
    recall = confusionMatrix[0] / (confusionMatrix[0] + confusionMatrix[3])
    f1 = 2 * precision * recall / (precision + recall)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import OneClassSVM

class termDocumentEmbedding:
    def __init__(self, texts):
        self.train(texts)

    def train(self, texts):
        self.count_vectorizer = CountVectorizer(analyzer=tokenize)
        self.train_counts = self.count_vectorizer.fit_transform(texts)
        self.clf = OneClassSVM()
        self.clf = self.clf.fit(self.train_counts)

    def classify(self, test_instance):
        test_count = self.count_vectorizer.transform([test_instance])
        return self.clf.predict(test_count)[0]

class BERTEmbedding:
    def __init__(self, training_embeddings):
        self.train(training_embeddings)

    def train(self, training_embeddings):
        self.clf = OneClassSVM().fit(training_embeddings)

    def classify(self, test_instance_embedding):
        return self.clf.predict(test_instance_embedding)[0]
    

if __name__ == "__main__":

    random.seed()
    AUTHOR_IN_QUESTION_FILENAME = 'blogs/3611601.male.17.indUnk.Leo.xml'

    print('Building Datasets')

    authorInQuestionTexts, authorInQuestionLabels = readFileData(AUTHOR_IN_QUESTION_FILENAME, 1)

    authorInQuestionMetadata = extractInfoFromFileName(AUTHOR_IN_QUESTION_FILENAME)

    test_texts = authorInQuestionTexts[200:len(authorInQuestionTexts)]
    test_labels = authorInQuestionLabels[200:len(authorInQuestionLabels)]
    train_texts = authorInQuestionTexts[0:200]

    testSetAuthorMetadata = []

    for i in range(len(test_texts)):
        testSetAuthorMetadata.append(authorInQuestionMetadata)

    test_texts, test_labels, testSetAuthorMetadata = buildTestSet(test_texts, 
                                                                  test_labels, 
                                                                  testSetAuthorMetadata,
                                                                  AUTHOR_IN_QUESTION_FILENAME
                                                                  )

    print(len(test_texts))
    print(len(test_labels))
    print(len(testSetAuthorMetadata))
    print(len(train_texts))

    print('Datasets Built')

    classifier = termDocumentEmbedding(train_texts)
    results = [classifier.classify(x) for x in test_texts]

    print('OUTPUTTING TD RESULTS')
    output_results(results)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    BERTModel = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    embedding_layer = BERTModel.bert.embeddings

    print('Tokenizing Training Data')

    training_tokens = preprocess_training_data(tokenizer, train_texts)

    print('Tokenizing complete.. Embedding Training Data')

    training_embeddings = embedding_layer(training_tokens['input_ids'])
    print(tf.shape(training_embeddings))

    print('Embedding Complete.. Flattening Training Embeddings')

    flattened_training_embeddings = tf.reshape(training_embeddings, 
                                                [len(training_embeddings), 
                                                    len(training_embeddings[0]) * len(training_embeddings[0][0])
                                                    ])

    print('Flattening Complete.. Training One Class SVM Model')
    print(tf.shape(flattened_training_embeddings))

    classifier = BERTEmbedding(flattened_training_embeddings)
        
    print('Classifying Test Instances')

    results = []
    for test_instance in test_texts:
        tokenized_test_instance = preprocess_test_data(tokenizer, test_instance)
        test_instance_embedding = embedding_layer(tokenized_test_instance['input_ids'])
        flattened_test_instance_embedding = tf.reshape(test_instance_embedding, [1, 
                                                                                     len(test_instance_embedding[0]) * 
                                                                                         len(test_instance_embedding[0][0])
                                                                                         ])
        results.append(classifier.classify(flattened_test_instance_embedding))

    print('OUTPUTTING BERT RESULTS')
    output_results(results)




    