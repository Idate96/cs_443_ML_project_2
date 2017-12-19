import pickle
import numpy as np

g = open('numpy.txt', 'w')
vocab = dict()
with open('glove.twitter.27B.200d.txt', 'r') as f:
    for idx, line in enumerate(f):
        index = line.index(' ')
        vocab[line[:index].strip()] = idx
        g.write(line[index:])

#print(vocab)
f.close()
g.close()
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

embeddings = np.genfromtxt('numpy.txt')
#print(embedding)
np.save('embeddings', embeddings)