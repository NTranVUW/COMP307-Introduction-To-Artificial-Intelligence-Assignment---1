The program is ran via the terminal: 

perceptron.py <data_file> <learning_rate> <train_size>(optional)

If train_size is not provided will just train the perceptron. 
Train_size is a float between 0.0-1.0. 
Specifying train_size will split the dataset train_size:1-train_size and train the perceptron and then test the perceptron 
using the test set with the training weights and bias.
   