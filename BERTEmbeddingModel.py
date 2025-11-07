import sys

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

    return texts, labels



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


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import OneClassSVM

class Baseline:
    def __init__(self, labels):
        self.train(labels)

    def train(self, labels):
        label_freqs = {}
        for label in labels:
            if label in label_freqs:
                label_freqs[label] += 1
            else:
                label_freqs[label] = 1
        self.mfc = sorted(label_freqs, reverse=True,
                          key=lambda x : label_freqs[x])[0]
        
    def classify(self, test_instance):
        return self.mfc
    
class oneClassSVM:
    def __init__(self, training_embeddings):
        self.train(training_embeddings)

    def train(self, training_embeddings):
        self.clf = OneClassSVM().fit(training_embeddings)

    def classify(self, test_instance_embedding):
        return self.clf.predict(test_instance_embedding)[0]
    
if __name__ == "__main__":

    method = sys.argv[1]

    print('Building Datasets')

    train_texts, train_labels = readFileData('blogs/78196.female.35.indUnk.Aries.xml', 1)

    test_texts = train_texts[len(train_texts)//2:len(train_texts)]
    train_texts = train_texts[0:len(train_texts)//2]
    print(len(train_texts))

    print('Datasets Built')

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

    if method == 'baseline':
        classifier = Baseline(train_labels)
        results = [classifier.classify(x) for x in test_texts] 

    if method == 'oneClassSVM':
        classifier = oneClassSVM(flattened_training_embeddings)
        
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

    posPreds = 0
    negPreds = 0

    with open('results/Experiment_19BERT.txt', 'w') as outfile:

        outfile.write('Experiment on File blogs/78196.female.35.indUnk.Aries.xml\n\n')

        for i in range(len(results)):
            outfile.write(str(results[i]) + '\n')
            if results[i] == 1:
                posPreds += 1
            else:
                negPreds += 1

    print(posPreds)
    print(negPreds)