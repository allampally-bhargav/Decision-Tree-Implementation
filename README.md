# Decision-Tree-Implementation
Implemented a fixed-depth decision tree algorithm


Problem: Implementation of a fixed-depth decision tree algorithm, that is, the input to the ID3 algorithm will include the training data and maximum depth of the tree to be learned.

Data Sets: The MONK’s Problems were the basis of a first international comparison of learning algorithms

The training and test files for the three problems are named monks-X.train and monks-X.test. There are six attributes/features (columns 2–7), and binary class labels (column 1).See monks.names for more details.

Visualization: The function render_dot_file() can be used to generate .png images of the trees learned by both scikit-learn and your code.



Glass Identification dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data

Used scikit-learn’s DecisionTreeClassifier to learn a decision tree using criterion='entropy' for depth = 1, 3, 5 for Monks data and Glass identification dataset and 
Studied scikit-learn’s confusion matrix() function.

Used my own implementation of decision tree and compared the results.
