from dataloader import *
from algorithms import *
import numpy as np
import scipy as sp
import scipy.stats as stat
import timeit

# dataset parameters
data_size = 40000
folds = 10
folds_icv = 4
max_words_in_sent = 12
word_embedding_size = 50
word_embedding_w2v_file = 'glove_6B_50d_w2v.txt'
num_runs = 10


def is_overlapping(lower1, upper1, lower2, upper2):
    return not (lower1 > upper2 or upper1 < lower2)


def mean_confidence_interval(data, confidence=0.95):
    # code source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0*np.array(data)
    n = len(a)
    mean, std_err = np.mean(a), stat.sem(a)
    h = std_err * sp.stats.t.ppf((1+confidence)/2., n-1)
    return mean, mean-h, mean+h


if __name__ == '__main__':
    start = timeit.default_timer()
    print('Loading data...')
    # values = load_binary_labelled_data('quora_duplicate_questions.tsv', 3, 4, 5, size=data_size)
    values = load_binary_labelled_data('quora_40000.txt', 0, 1, 2, ignore_header=False)
    # import nltk
    # import statistics as statis
    # max_global = 0
    # lens = []
    # for v in values:
    #     l1 = len(nltk.word_tokenize(v[0]))
    #     l2 = len(nltk.word_tokenize(v[0]))
    #     lens.append(l1)
    #     lens.append(l2)
    #     max_words = max(l1, l2)
    #     max_global = max(max_words, max_global)
    # print('max num words in q: ' + str(max_global))
    # print('mean num words' + str(statis.mean(lens)))
    # print('median num words' + str(statis.median(lens)))
    # print('mode num words' + str(statis.mode(lens)))
    print(str(len(values)) + ' instances loaded')
    print('Generating vector representation using Glove6B, word embedding size ' + str(word_embedding_size) + '...')
    vec_values = gen_vector_rep_for_sentences(values, word_embedding_w2v_file, word_embedding_size, max_words_in_sent)
    stop = timeit.default_timer()
    print('Done, time taken='+str(stop-start)+' seconds')
    lr_f1s = []
    nb_f1s = []
    dt_f1s = []
    lr_acc = []
    nb_acc = []
    dt_acc = []
    for run in range(num_runs):
        k_folds = split_binary_data_kfolds(vec_values, folds, label_idx=1)
        print('run='+str(run+1)+', external CV folds=' + str(folds) + ' fold size=' + str(len(k_folds[0])))
        # training/testing with external k folds CV
        print('Init training with ' + str(folds_icv) + ' folds internal CV...')
        params_tuning = True
        for i in range(folds):
            test_mat, test_truth_labels = gen_input_matrix(k_folds[i])
            training_data = combine_folds(k_folds, [i])

            print('Training Logistic Regression...')
            # train logistic regression classifier
            logistic_regression = train_logistic_reg(training_data, folds_icv, params_tuning=params_tuning)
            # test logistic regression classifier
            scores = test_logistic_reg(logistic_regression, test_mat, test_truth_labels)
            lr_f1s.append(scores[0])
            lr_acc.append(scores[1])
            print('F1 score for Logistic Regression: ' + str(scores[0]))
            print('Accuracy score for Logistic Regression: ' + str(scores[1]))

            print('Training Naive Bayes...')
            # train naive bayers classifier
            naive_bayes = train_naive_bayes(training_data, folds_icv, params_tuning=params_tuning)
            # test naive bayers classifier
            scores = test_naive_bayes(naive_bayes, test_mat, test_truth_labels)
            nb_f1s.append(scores[0])
            nb_acc.append(scores[1])
            print('F1 score for GaussianNB: ' + str(scores[0]))
            print('Accuracy score for GaussianNB: ' + str(scores[1]))

            print('Training Decision Tree...')
            # train DT classifier
            tree = train_decision_tree(training_data, folds_icv, params_tuning=params_tuning)
            # test DT classifier
            scores = test_decision_tree(tree, test_mat, test_truth_labels)
            dt_f1s.append(scores[0])
            dt_acc.append(scores[1])
            print('F1 score for Decision Tree: ' + str(scores[0]))
            print('Accuracy score for Decision Tree: ' + str(scores[1]))

    # cache results
    cache_list_f1 = []
    cache_list_acc = []
    for i in range(len(lr_f1s)):
        cache_list_f1.append([lr_f1s[i], nb_f1s[i], dt_f1s[i]])
        cache_list_acc.append([lr_acc[i], nb_acc[i], dt_acc[i]])
    write_data_to_file(cache_list_f1, 'f1_lr_nb_dt_25.txt')
    write_data_to_file(cache_list_acc, 'acc_lr_nb_dt_25.txt')
    # confidence intervals
    lr_m_f1, lr_lower_f1, lr_upper_f1 = mean_confidence_interval(lr_f1s, 0.95)
    nb_m_f1, nb_lower_f1, nb_upper_f1 = mean_confidence_interval(nb_f1s, 0.95)
    dt_m_f1, dt_lower_f1, dt_upper_f1 = mean_confidence_interval(dt_f1s, 0.95)
    lr_m_acc, lr_lower_acc, lr_upper_acc = mean_confidence_interval(lr_acc, 0.95)
    nb_m_acc, nb_lower_acc, nb_upper_acc = mean_confidence_interval(nb_acc, 0.95)
    dt_m_acc, dt_lower_acc, dt_upper_acc = mean_confidence_interval(dt_acc, 0.95)
    print('mean f1 for logistic regression: ' + str(lr_m_f1))
    print('mean f1 for naive bayes: ' + str(nb_m_f1))
    print('mean f1 for decision tree: ' + str(dt_m_f1))
    print('mean acc for logistic regression: ' + str(lr_m_acc))
    print('mean acc for naive bayes: ' + str(nb_m_acc))
    print('mean acc for decision tree: ' + str(dt_m_acc))
    print('f1 confidence interval for logistic regression: ' + str(lr_lower_f1) + ' ' + str(lr_upper_f1))
    print('f1 confidence interval for naive bayes: ' + str(nb_lower_f1) + ' ' + str(nb_upper_f1))
    print('f1 confidence interval for decision tree: ' + str(dt_lower_f1) + ' ' + str(dt_upper_f1))
    print('acc confidence interval for logistic regression: ' + str(lr_lower_acc) + ' ' + str(lr_upper_acc))
    print('acc confidence interval for naive bayes: ' + str(nb_lower_acc) + ' ' + str(nb_upper_acc))
    print('acc confidence interval for decision tree: ' + str(dt_lower_acc) + ' ' + str(dt_upper_acc))
    if is_overlapping(lr_lower_f1, lr_upper_f1, nb_lower_f1, nb_upper_f1):
        print('logistic regression and naive bayes f1 confidence intervals overlap')
    else:
        print('logistic regression and naive bayes f1 confidence intervals do not overlap')
    if is_overlapping(lr_lower_f1, lr_upper_f1, dt_lower_f1, dt_upper_f1):
        print('logistic regression and decision tree f1 confidence intervals overlap')
    else:
        print('logistic regression and decision tree f1 confidence intervals do not overlap')
    if is_overlapping(nb_lower_f1, nb_upper_f1, dt_lower_f1, dt_upper_f1):
        print('naive bayes and decision tree f1 confidence intervals overlap')
    else:
        print('naive bayes and decision tree f1 confidence intervals do not overlap')

    if is_overlapping(lr_lower_acc, lr_upper_acc, nb_lower_acc, nb_upper_acc):
        print('logistic regression and naive bayes f1 confidence intervals overlap')
    else:
        print('logistic regression and naive bayes f1 confidence intervals do not overlap')
    if is_overlapping(lr_lower_acc, lr_upper_acc, dt_lower_acc, dt_upper_acc):
        print('logistic regression and decision tree f1 confidence intervals overlap')
    else:
        print('logistic regression and decision tree f1 confidence intervals do not overlap')
    if is_overlapping(nb_lower_acc, nb_upper_acc, dt_lower_acc, dt_upper_acc):
        print('naive bayes and decision tree f1 confidence intervals overlap')
    else:
        print('naive bayes and decision tree f1 confidence intervals do not overlap')
    # anova test
    f_val_f1, p_val_f1 = stat.f_oneway(lr_f1s, nb_f1s, dt_f1s)
    print('f1 p val = ' + str(p_val_f1))

    f_val_acc, p_val_acc = stat.f_oneway(lr_acc, nb_acc, dt_acc)
    print('acc p val = ' + str(p_val_acc))
