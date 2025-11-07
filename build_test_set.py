from multiAuthorTestSetModels import extractInfoFromFileName, readFileData
import random, os

random.seed()

def buildTestSet(test_set, test_labels, testSetAuthorMetadata):
    for file in os.listdir('blogs'):
        if random.random() < 0.05:
            fileName = 'blogs/' + file
            filePosts, fileLabels = readFileData(fileName, 0)
            metadata = extractInfoFromFileName(fileName)
            for i in range(len(filePosts)):
                if random.random() < (1/len(filePosts)):
                    test_set.append(filePosts[i])
                    test_labels.append(fileLabels[i])
                    testSetAuthorMetadata.append(metadata)

    return test_set, test_labels, testSetAuthorMetadata