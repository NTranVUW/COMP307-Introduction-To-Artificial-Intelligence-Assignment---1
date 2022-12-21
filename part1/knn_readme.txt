The program is ran via the terminal: 

python knn.py <training data> <test data> <k> <distance metric>

where k can be any integer greater than 0, distance_metric is either euclidean or manhattan (all lowercase) because other 
distance functions are hard. 

The program is dataset agnostic and theoretically can be used to classify any dataset that is space delimitted and 
has the last feature column as it's classes. The program also does not require any libraries outside of base python, 
the only libraries used is sys for command line arguments,  operator for dictionary sorting and math for square root and 
other mathematics opperators. 