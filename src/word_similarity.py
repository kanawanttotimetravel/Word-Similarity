import numpy as np
import pickle
from tqdm import tqdm

from pathlib import Path

from scipy.stats import pearsonr, spearmanr

# The source code file is located in the /src folder
DIRECTORY = Path(__file__).parent.parent.resolve()

def read_pretrain_embeds():
    """Read the pre-trained embeddings from the file and save it as a dictionary
    """
    with open(DIRECTORY / 'word2vec' / 'W2V_150.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()
    
    word_count = int(data[0])
    word_dim = int(data[1])

    data = data[2:]

    word2vec = {}

    for line in data:
        line = line.split()
        word = line[0]
        vec = np.array([float(x) for x in line[1:]])
        word2vec[word] = vec

    with open(DIRECTORY / 'word2vec' / 'word2vec.pkl', 'wb') as file:
        pickle.dump(word2vec, file)


def cosine_similarity(word2vec, word1, word2):
    """Calculate the cosine similarity between two words based on an embeddings dictionary

    Args:
        word2vec (dict): Embedding dictionary
        word1 (str): 
        word2 (str): 

    Returns:
        float: The cosine similarity
    """
    # In case one of the words is not in the dictionary
    if word1 not in word2vec:
        syllables = word1.split('_')
        abnormal = True
        for syllable in syllables:
            if syllable in word2vec:
                v1 = word2vec[syllable]
                abnormal = False
                break
        if abnormal:
            return 0.5
    else:
        v1 = word2vec[word1]

    if word2 not in word2vec:
        syllables = word2.split('_')
        abnormal = True
        for syllable in syllables:
            if syllable in word2vec:
                v2 = word2vec[syllable]
                abnormal = False
                break
        if abnormal:
            return 0.5
    else:
        v2 = word2vec[word2]

    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return sim

def test():
    with open(DIRECTORY / 'word2vec' / 'word2vec.pkl', 'rb') as file:
        word2vec = pickle.load(file)

    with open(DIRECTORY / 'datasets' / 'ViSim-400' / 'ViSim-400.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()
    data = data[1:]
    results = []
    labels = []
    for line in tqdm(data):
        line = line.split()
        word1 = line[0]
        word2 = line[1]
        sim = line[4]

        labels.append(float(sim))
        results.append(cosine_similarity(word2vec, word1, word2))

    # These lines are to display the result of each pair

    # open(DIRECTORY / 'src' / 'output.txt', 'w').close()
    # with open(DIRECTORY / 'src' / 'output.txt', 'a', encoding='utf-8') as file:
    #     for i, result in enumerate(results):
    #         line = data[i].split()
    #         word1 = line[0]
    #         word2 = line[1]
    #         file.write(f'{word1} {word2} {result} {labels[i]}\n') 
    

    print("Pearson correlation coefficient: ", pearsonr(labels, results))
    print("Spearman's rank correlation coefficient: ", spearmanr(labels, results))


if __name__ == "__main__":
    # Run this function if it hasn't been run once 
    # read_pretrain_embeds()
    test()