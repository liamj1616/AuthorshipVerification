import sys, re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

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
    def __init__(self, texts, labels):
        self.train(texts, labels)

    def train(self, texts, labels):
        self.count_vectorizer = CountVectorizer(analyzer=tokenize)
        self.train_counts = self.count_vectorizer.fit_transform(texts)
        self.clf = OneClassSVM()
        self.clf = self.clf.fit(self.train_counts)

    def classify(self, test_instance):
        test_count = self.count_vectorizer.transform([test_instance])
        return self.clf.predict(test_count)[0]
    
if __name__ == "__main__":

    method = sys.argv[1]

    train_texts, train_labels = readFileData('blogs/78196.female.35.indUnk.Aries.xml', 1)

    test_texts = train_texts[len(train_texts)//2:len(train_texts)]
    train_texts = train_texts[0:len(train_texts)//2]
    print(len(test_texts))

    if method == 'baseline':
        classifier = Baseline(train_labels)

    if method == 'oneClassSVM':
        classifier = oneClassSVM(train_texts, train_labels)

    results = [classifier.classify(x) for x in test_texts]

    with open('results/Experiment_19Count.txt', 'w') as outfile:

        outfile.write('Experiment on File blogs/78196.female.35.indUnk.Aries.xml\n\n')
        posPreds = 0
        negPreds = 0
        for i in range(len(results)):
            outfile.write(str(results[i]) + '\n')
            if results[i] == 1:
                posPreds += 1
            else:
                negPreds += 1

    print(posPreds)
    print(negPreds)
    print(posPreds/len(test_texts))

    
