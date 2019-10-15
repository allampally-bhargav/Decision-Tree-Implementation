# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import metrics


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    uniqueval=np.unique(x)
    
    dict={i: [] for i in uniqueval}
    for j,k in enumerate(x):
        dict[k].append(j)
   # print(dict)
    return dict
        
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    val, count=np.unique(y,return_counts=True)
    entropy=0
    sum1 = np.sum(count)
    for i in range(len(val)):
        total=np.sum([(-count[i]/sum1)*np.log2(count[i]/sum1)])
        entropy+=total
    #print(entropy)
    return entropy
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    hy=entropy(y)
    

    partitionX= partition(x)
    total_samples= len(x)
    hyx=0
    for i in partitionX.keys():
        px= (float) (len(partitionX[i])/ total_samples)
        new_y = [y[i] for i in partitionX[i]]
        hy_x = entropy(new_y)
        hyx += (px * hy_x)
   # print(hy-hyx)
    return (hy - hyx)
        
    
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    
    root={}
    
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range (len(x[0])):
            for j in np.unique(np.array([elem[i] for elem in x])):
                attribute_value_pairs.append((i, j))
    attribute_value_pairs= np.array(attribute_value_pairs)
    
    yval, ycount = np.unique(y, return_counts=True)
    if len(yval)==1:
        return yval[0]
    
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return yval[np.argmax(ycount)]
    
    infoGain = []
    
    for i , val in attribute_value_pairs:
        infoGain.append(mutual_information(np.array((x[:, i]== val).astype(int)),y))
        
    infoGain = np.array(infoGain)
    (i,val) = attribute_value_pairs[np.argmax(infoGain)]
    
    new_partition = partition(np.array((x[:,i]==val).astype(int)))
    
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argmax(infoGain), 0 )
    
    for value, index in new_partition.items():
        set_x = x.take(np.array(index), axis=0)
        set_y = y.take(np.array(index), axis=0)
        decision = bool(value)
        
        root[(i,val,decision)]=id3(set_x, set_y, attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
   # print(root)
    return root
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    for split_val, subtree in tree.items():
        index = split_val[0]
        value = split_val[1]
        split_decision = split_val[2]
        
        if split_decision ==(x[index]==value):
            if type(subtree) is dict:
                label = predict_example(x, subtree)
            else:
                label = subtree
           # print(label)
            return label
        
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    
    l = len(y_true)
    #print(np.sum(np.absolute(y_true - y_pred))/l)
    return np.sum(np.absolute(y_true - y_pred))/l
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    y_trn_pred = [predict_example(x, decision_tree) for x in Xtrn]
    trn_err = compute_error(ytrn, y_trn_pred)
    print('Training Error = {0:4.2f}%.'.format(trn_err * 100))
    
    #Question b monk 1
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    tst_err1 = []
    trn_err1 = []
    for val in range(1, 11):
        decision_tree = id3(Xtrn, ytrn, max_depth=val)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        tst_err1.append(tst_err * 100)
        y_trn_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_trn_pred)
        trn_err1.append(trn_err * 100)
    print('Avg test error')
    print(tst_err1)
    print('Avg training error')
    print(trn_err1)

    X_test = [idx for idx in range(1, 11)]
    Y_test = tst_err1
    X_trn = [i for i in range(1, 11)]
    Y_trn = trn_err1
    plt.plot(X_test, Y_test, linewidth=1.0)
    plt.plot(X_trn, Y_trn, linewidth=1.0)
    plt.title("Monks 1 Training, Test error vs Depth")
    plt.xlabel('Depth')
    plt.ylabel('Test error - blue , Train error - red')
    
    plt.show()
    
    #Question b monk 2
    # Load the training data
    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    tst_err2 = []
    trn_err2 = []
    for val in range(1, 11):
        decision_tree = id3(Xtrn, ytrn, max_depth=val)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        tst_err2.append(tst_err * 100)
        y_trn_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_trn_pred)
        trn_err2.append(trn_err * 100)
    print('Avg test error')
    print(tst_err2)
    print('Avg training error')
    print(trn_err2)

    X_test = [idx for idx in range(1, 11)]
    Y_test = tst_err2
    X_trn = [i for i in range(1, 11)]
    Y_trn = trn_err2
    plt.plot(X_test, Y_test, linewidth=1.0)
    plt.plot(X_trn, Y_trn, linewidth=1.0)
    plt.title("Monks 2 Training, Test error vs Depth")
    plt.xlabel('Depth')
    plt.ylabel('Test error - blue , Train error - red')
    
    plt.show()
    
    #Question b monk 3
    # Load the training data
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    tst_err3 = []
    trn_err3 = []
    for val in range(1, 11):
        decision_tree = id3(Xtrn, ytrn, max_depth=val)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        tst_err3.append(tst_err * 100)
        y_trn_pred = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_trn_pred)
        trn_err3.append(trn_err * 100)
    print('Avg test error')
    print(tst_err3)
    print('Avg training error')
    print(trn_err3)

    X_test = [idx for idx in range(1, 11)]
    Y_test = tst_err3
    X_trn = [idx for idx in range(1, 11)]
    Y_trn = trn_err3
    plt.plot(X_test, Y_test, linewidth=1.0)
    plt.plot(X_trn, Y_trn, linewidth=1.0)
    plt.title("Monks 3 Training, Test error vs Depth")
    plt.xlabel('Depth')
    plt.ylabel('Test error - blue , Train error - red')
    
    plt.show()
    
    
    #Question c
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    for d in range(1,6,2):
        
        decision_tree = id3(Xtrn, ytrn, max_depth=d)

        # Pretty print it to console
        pretty_print(decision_tree)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './my_learned_tree_depth_{}'.format(d))
        
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        con_matrix = confusion_matrix(ytst, y_pred)
        print("confusion matrix for depth_{}".format(d))
        print(con_matrix)
        
    
    #Question d 
     #d.scikit-learn entropy criterion
    col_head = ['y', 'x1', 'x2', 'x3', 'x4', 'x5']
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    
    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:] 
    
    for d in range(1, 6, 2):
        classifier = DecisionTreeClassifier(max_depth=d,criterion='entropy')
        classifier = classifier.fit(Xtrn, ytrn)
        
        y_pred = classifier.predict(Xtst)
        
        con_matrix = confusion_matrix(ytst, y_pred)
        print("confusion matrix of scikit-learn's for depth>>{}".format(d))
        print(con_matrix)
        
        dot_data = StringIO()
        export_graphviz(classifier, out_file=dot_data,
                        filled=True, rounded=True,special_characters=True, feature_names=col_head, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('mydecision_scikit_entropy{}.png'.format(d))
        
        Image(graph.create_png())
        
    #Question e
    col_head = ['y', 'x1', 'x2', 'x3', 'x4', 'x5','x6','x7','x8','x9']
    # Load the training data
    M = np.genfromtxt('./glass.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # Load the test data
    M = np.genfromtxt('./glass.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    for d in range(1,6,2):
        
        decision_tree = id3(Xtrn, ytrn, max_depth=d)

        # Pretty print it to console
        pretty_print(decision_tree)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './my_learned_tree_glass_depth_{}'.format(d))
        
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        con_matrix = confusion_matrix(ytst, y_pred)
        print("confusion matrix for depth_{}".format(d))
        print(con_matrix)
        
    
    for d in range(1, 6, 2):
        classifier = DecisionTreeClassifier(max_depth=d,criterion='entropy')
        classifier = classifier.fit(Xtrn, ytrn)
        
        y_pred = classifier.predict(Xtst)
        
        con_matrix = confusion_matrix(ytst, y_pred)
        print("confusion matrix of scikit-learn's for depth>>{}".format(d))
        print(con_matrix)
        
        dot_data = StringIO()
        export_graphviz(classifier, out_file=dot_data,
                        filled=True, rounded=True,special_characters=True, feature_names=col_head, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('mydecision_scikit_glass_entropy{}.png'.format(d))
        
        Image(graph.create_png())