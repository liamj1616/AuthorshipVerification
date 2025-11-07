import matplotlib.pyplot as plt

numPosts = [
    1052, 1018, 930, 844, 733, 682, 667, 648, 
    642, 627, 561, 511, 503, 473, 390, 300
]
accuraciesBERT = [
    0.54, 0.20, 0.73, 0.67, 0.5, 0.54, 0.41, 0.91,
    0.42, 0.39, 0.79, 0.34, 0.46, 0.79, 0.59, 0.45
]
accuraciesTD = [
    0.64, 0.29, 0.48, 0.58, 0.54, 0.57, 0.34, 0.21,
    0.49, 0.46, 0.42, 0.48, 0.44, 0.57, 0.59, 0.56
]

plt.scatter(numPosts, accuraciesBERT, label='BERT')
plt.scatter(numPosts, accuraciesTD, label='Term-document')
plt.legend()
plt.xlabel('Number of Blog Posts')
plt.ylabel('Accuracies')
plt.title('Experiment 1 Results')


for i in range(len(numPosts)):
    x = [numPosts[i], numPosts[i]]
    y = [accuraciesBERT[i], accuraciesTD[i]]
    plt.plot(x, y)

plt.savefig('plots/Experiment1Results.png')
