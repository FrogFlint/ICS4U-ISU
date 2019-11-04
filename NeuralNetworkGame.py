import numpy as np



def sigmoid(x):
    return x / (abs(x) + 1)
def normalizedSigmoid(x):
    return (sigmoid(2*x) + 1) / 2

class NeuralNetwork:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections
    def evaluate(self, inputData, outputFilter):
        currentLayer = normalizedSigmoid(inputData + self.nodes[0])
        for currentNodes, currentConnections in zip(self.nodes[1:], self.connections):
            currentLayer = normalizedSigmoid((currentConnections @ currentLayer) + currentNodes)
        return currentLayer




networks = []
dataFile = open(r"TrainingData.txt", "r")  # w+

data = dataFile.read().split("\n\n")
for item in data:
    item = item.split("\n")
    hiddenLayers = int((len(item) - 1) / 2) - 1

    nodes = [np.array(x.split(","), dtype = "float32") for x in item[:hiddenLayers + 2]]
    connections = [np.array(x.split(","), dtype = "float32").reshape(int(len(nodes[i])), int(len(nodes[i + 1]))) for i,x in enumerate(item[hiddenLayers + 2:])]
    networks.append(NeuralNetwork(nodes, connections))


for network in networks:
    print("nodes:\n\t" + str(network.nodes))
    print("connections:\n\t" + str(network.connections))
    print("\n")

print(networks[2].evaluate([0]*4, [0]*4))

dataFile.close()

