# TransE-Practice
Modification of the TransE code taken from GitHub repository for training and testing the model on FB15k dataset.


All the code and datasets have been taken from: https://github.com/thunlp/TensorFlow-TransX

The provided code basically does the following tasks:
1) Train a TransE based model to learn latent embeddings of FB15k data set knowledge graph.
2) Use the testing data to check performance of the trained model.

The model parameters can be changed in the Config class to suit current needs.

All data required to train the model is present in the data folder. For testing, the test.txt file in data had to be converted to id format.For this, you can refer to the code getTest2id.py in the FB15k folder. This code converts the test.txt file to an id format which can be easily used for testing of the model.

Once the training loop completes,tensorflow saves the model using the line "saver.sess(...)". Hence you donot need to train everytime you run the code.For testing, the saved model can be directly loaded using "saver.restore(...)". 

EVALUATION:
For testing, 3 evalaution matrices have been used.
1) Hits@1 2) Hits@5 3) Hits@10
These basically tell if the actual relation present in a triple of test data is top 1/5/10 predicitons sorted in ascending order of the score of triple.
