from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.classification import f1_score, accuracy_score
from dataloader import *


def train_logistic_reg(training_data, folds_icv, params_tuning=True):
    # hyper-parameters to tune
    # smaller strength = stronger regularization
    inverse_reg_strength = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0]
    regularizer = ['l1', 'l2']
    err_tol = [1e-4]
    # use k folds internal CV to find the best parameters
    folds_train = split_binary_data_kfolds(training_data, folds_icv, label_idx=1)
    # best params
    curr_highest_score = None
    best_reg_strength = 0.05
    best_reg = 'l1'
    best_tol = 1e-4
    if params_tuning:
        for strength in inverse_reg_strength:
            print('C ' + str(strength))
            for reg in regularizer:
                print('reg ' + reg)
                for tol in err_tol:
                    print('tol ' + str(tol))
                    f1_scores = []
                    for j in range(len(folds_train)):
                        validation_mat, validation_truth_labels = gen_input_matrix(folds_train[j])
                        train_mat, train_truth_labels = gen_input_matrix(combine_folds(folds_train, [j]))
                        model = LogisticRegression(penalty=reg, C=strength, tol=tol)
                        model.fit(train_mat, train_truth_labels)
                        scores = test_logistic_reg(model, validation_mat, validation_truth_labels)
                        f1_scores.append(scores[0])
                    # select the parameter that gives the best avg f1 score among k folds
                    if curr_highest_score is None or curr_highest_score < sum(f1_scores)/float(len(f1_scores)):
                        curr_highest_score = sum(f1_scores)/float(len(f1_scores))
                        best_reg_strength = strength
                        best_reg = reg
                        best_tol = tol
        print('Best parameter for Logistic Regression: penalty='+best_reg+' C='+str(best_reg_strength)+
              ' tol='+str(best_tol))
    # train on entire training set after finding the best parameters
    all_data, all_labels = gen_input_matrix(combine_folds(folds_train, []))
    model = LogisticRegression(penalty=best_reg, C=best_reg_strength, tol=best_tol)
    model.fit(all_data, all_labels)
    return model


def test_logistic_reg(model, test_data, truth_labels):
    predicted_labels = model.predict(test_data)
    rv = [f1_score(truth_labels, predicted_labels), accuracy_score(truth_labels, predicted_labels)]
    return rv


def train_naive_bayes(training_data, folds_icv, params_tuning=True):
    class_priors = [None, np.array([0.5, 0.5])]
    # use k folds internal CV to find the best parameters
    folds_train = split_binary_data_kfolds(training_data, folds_icv, label_idx=1)
    # best params
    curr_highest_score = None
    best_prior = None
    if params_tuning:
        for prior in class_priors:
            f1_scores = []
            for j in range(len(folds_train)):
                validation_mat, validation_truth_labels = gen_input_matrix(folds_train[j])
                train_mat, train_truth_labels = gen_input_matrix(combine_folds(folds_train, [j]))
                model = GaussianNB(priors=prior)
                model.fit(train_mat, train_truth_labels)
                scores = test_naive_bayes(model, validation_mat, validation_truth_labels)
                f1_scores.append(scores[0])
            if curr_highest_score is None or curr_highest_score < sum(f1_scores) / float(len(f1_scores)):
                curr_highest_score = sum(f1_scores) / float(len(f1_scores))
                best_prior = prior
        print('Best parameter for GaussianNB: prior=' + str(best_prior))
    all_data, all_labels = gen_input_matrix(combine_folds(folds_train, []))
    model = GaussianNB(priors=best_prior)
    model.fit(all_data, all_labels)
    return model


def test_naive_bayes(model, test_data, truth_labels):
    predicted_labels = model.predict(test_data)
    rv = [f1_score(truth_labels, predicted_labels), accuracy_score(truth_labels, predicted_labels)]
    return rv


def train_SVM(training_data, folds_icv, params_tuning=True):
    inverse_reg_strength = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    probs = [False, True]  # probability estimates
    shrinks = [True, False]  # shrinking heuristics
    folds_train = split_binary_data_kfolds(training_data, folds_icv, label_idx=1)
    # best params
    curr_highest_score = None
    best_strength = 1.0
    best_kernel = 'rbf'
    best_prob = False
    best_shrink = True
    if params_tuning:
        for strength in inverse_reg_strength:
            print('svm str')
            for k in kernels:
                print('svm kernel')
                for p in probs:
                    print('svm prob')
                    for s in shrinks:
                        print('svm shrink')
                        f1_scores = []
                        for j in range(len(folds_train)):
                            validation_mat, validation_truth_labels = gen_input_matrix(folds_train[j])
                            train_mat, train_truth_labels = gen_input_matrix(combine_folds(folds_train, [j]))
                            model = SVC(C=strength, kernel=k, probability=p, shrinking=s)
                            model.fit(train_mat, train_truth_labels)
                            scores = test_SVM(model, validation_mat, validation_truth_labels)
                            f1_scores.append(scores[0])
                        if curr_highest_score is None or curr_highest_score < sum(f1_scores) / float(len(f1_scores)):
                            curr_highest_score = sum(f1_scores) / float(len(f1_scores))
                            best_strength = strength
                            best_kernel = k
                            best_prob = p
                            best_shrink = s
        print('Best parameter for SVM: C=' + str(best_strength)+' kernel=' + str(best_kernel)
              + ' probability=' + str(best_prob)+' shrinking=' + str(best_shrink))
    all_data, all_labels = gen_input_matrix(combine_folds(folds_train, []))
    model = SVC(C=best_strength, kernel=best_kernel, probability=best_prob, shrinking=best_shrink)
    model.fit(all_data, all_labels)
    return model


def test_SVM(model, test_data, truth_labels):
    predicted_labels = model.predict(test_data)
    rv = [f1_score(truth_labels, predicted_labels), accuracy_score(truth_labels, predicted_labels)]
    return rv


def train_decision_tree(training_data, folds_icv, params_tuning=True):
    criterions = ['gini', 'entropy']
    splitters = ['best',
                 # 'random'
                 ]
    max_features = [
        None,
        0.005, 0.01, 0.25,
        0.5,
        0.75, 0.9,
        0.95,
        'sqrt', 'log2'
        ]
    max_depths = [
        None,
        5, 10, 20,
        50, 100
        ]
    folds_train = split_binary_data_kfolds(training_data, folds_icv, label_idx=1)
    # best params
    curr_highest_score = None
    best_criterion = 'entropy'
    best_splitter = 'best'
    best_max_f = None
    best_max_depth = 10
    if params_tuning:
        for c in criterions:
            print('decision tree criterions '+str(c))
            for s in splitters:
                print('decision tree splitters '+str(s))
                for f in max_features:
                    print('decision tree max_features '+str(f))
                    for d in max_depths:
                        print('decision tree max_depth ' + str(d))
                        f1_scores = []
                        for j in range(len(folds_train)):
                            validation_mat, validation_truth_labels = gen_input_matrix(folds_train[j])
                            train_mat, train_truth_labels = gen_input_matrix(combine_folds(folds_train, [j]))
                            model = DecisionTreeClassifier(criterion=c, splitter=s, max_features=f)
                            model.fit(train_mat, train_truth_labels)
                            scores = test_decision_tree(model, validation_mat, validation_truth_labels)
                            f1_scores.append(scores[0])
                        if curr_highest_score is None or curr_highest_score < sum(f1_scores) / float(len(f1_scores)):
                            curr_highest_score = sum(f1_scores) / float(len(f1_scores))
                            best_criterion = c
                            best_splitter = s
                            best_max_f = f
                            best_max_depth = d
        print('Best parameter for decision tree: criterion=' + str(best_criterion)+' splitter=' + str(best_splitter)
              + ' max_features=' + str(best_max_f) + ' max_depth=' + str(best_max_depth))
    all_data, all_labels = gen_input_matrix(combine_folds(folds_train, []))
    model = DecisionTreeClassifier(criterion=best_criterion, splitter=best_splitter, max_features=best_max_f,
                                   max_depth=best_max_depth)
    model.fit(all_data, all_labels)
    return model


def test_decision_tree(model, test_data, truth_labels):
    predicted_labels = model.predict(test_data)
    rv = [f1_score(truth_labels, predicted_labels), accuracy_score(truth_labels, predicted_labels)]
    return rv
