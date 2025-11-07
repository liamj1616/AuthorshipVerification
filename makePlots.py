import matplotlib.pyplot as plt

with open('results/Experiment_4Count.txt', 'r') as infile:
    header = infile.readline()
    header1 = infile.readline()

    normals = [0]
    anomalies = [0]

    for strRes in infile.readlines():
        res = int(strRes.strip())
        if res == 1:
            normals.append(normals[-1] + 1)
            anomalies.append(anomalies[-1])
        else:
            normals.append(normals[-1])
            anomalies.append(anomalies[-1] + 1)


plt.plot(anomalies, label='anomalies')
plt.plot(normals, label='normals')
plt.title('Anomaly Detection Over Time')
plt.xlabel('Time')
plt.ylabel('Counts')
plt.legend()
plt.savefig('plots/Experiment_4CountPlot.png')
