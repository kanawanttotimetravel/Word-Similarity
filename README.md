# PROGRAMMING ASSIGNMENT 1:  WORD SIMILARITY AND SEMANTIC RELATION CLASSIFICATION

Problem description: https://github.com/NLP-Projects/Word-Similarity 

## Prerequisites
- Download the pretrained embedding at `word2vec/link` and store at `word2vec/W2V_150.txt`

## Implementation
- The `cosine_similarity()` is implemented using this formula:
$$
    cosine\_similarity(v_1, v_2) = \frac{v_1 
    \cdot v_2}{\|v_1\| \cdot \|v_2\|}
$$


- `read_pretrain_embeds` read the embed file then store it into a binary file using `pickle` for quicker reading. This function only have to run **once**

- `test()` test the function using the Pearson correlation and Spearman rank correlation

## Results 
- Pearson: 0.43698374385268623
- Spearman: 0.3964172917677552