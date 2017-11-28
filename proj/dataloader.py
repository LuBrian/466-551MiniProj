import numpy as np


def write_data_to_file(datalist, filename, separator='\t'):
    f = open(filename, 'w', encoding='utf-8')
    for d in datalist:
        f.write(separator.join(str(e) for e in d))
        f.write('\n')
    f.close()


def gen_input_matrix(data_list):
    data = np.zeros((len(data_list), data_list[0][0].shape[1]))
    labels = []
    for idx in range(len(data_list)):
        data[idx, :] = data_list[idx][0]
        labels.append(data_list[idx][1])
    assert data.shape[0] == len(labels)
    return data, np.array(labels)


def combine_folds(folds, exclude_idx):
    rv = []
    for idx in range(len(folds)):
        if idx not in exclude_idx:
            rv += folds[idx]
    return rv


def gen_vector_rep_for_sentences(data, word_embedding_w2v_file, word_embedding_size, max_words_in_sent,
                                 vec_type='glove'):
    rv = np.zeros((len(data), 2 * max_words_in_sent * word_embedding_size))
    if vec_type == 'glove':
        import gensim
        import nltk
        glove = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_w2v_file, binary=False)
        for idx in range(len(data)):
            d = data[idx]
            sent_vec = np.zeros((max_words_in_sent, word_embedding_size))
            words = nltk.word_tokenize(d[0])
            for word_idx in range(min(len(words), max_words_in_sent)):
                word = words[word_idx].lower()
                if word in glove.vocab:
                    sent_vec[word_idx, :] = glove[word]
            tmp = sent_vec.flatten()
            sent_vec = np.zeros((max_words_in_sent, word_embedding_size))
            words = nltk.word_tokenize(d[1])
            for word_idx in range(min(len(words), max_words_in_sent)):
                word = words[word_idx].lower()
                if word in glove.vocab:
                    sent_vec[word_idx, :] = glove[word]
            tmp = tmp.reshape(1, -1)
            sent_vec = sent_vec.flatten().reshape(1, -1)
            rv[idx, :] = np.c_[tmp, sent_vec]
        return [[rv[i, :].reshape(1, -1), data[i][2]] for i in range(len(data))]
    else:
        print('Unknown vector representation type ' + str(type))
        exit(1)


def load_binary_labelled_data(file_path, sent1_col_idx, sent2_col_idx, label_idx,
                              ignore_header=True, shuffle=True, separator='\t', size=None):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    if ignore_header:
        lines = lines[1:]
    values = []
    for s in lines:
        split = s.split(separator)
        values.append([split[sent1_col_idx].rstrip(), split[sent2_col_idx].rstrip(), int(split[label_idx].rstrip())])
    if shuffle:
        np.random.shuffle(values)
    if size is not None:
        # take only a subset of the data, returned data size might be less than specificed size
        all_positive = [v for v in values if int(v[2]) == 1]
        all_negative = [v for v in values if int(v[2]) == 0]
        # try to maintain a 50:50 positive to negative ratio
        values = all_positive[:min(int(size/2), len(all_positive))]
        values += all_negative[:min(int(size/2), len(all_negative))]
        np.random.shuffle(values)
    # write_data_to_file(values)
    return values


def split_binary_data_kfolds(values, folds, label_idx=0):
    all_positive = [v for v in values if int(v[label_idx]) == 1]
    all_negative = [v for v in values if int(v[label_idx]) == 0]
    pos_ratio = float(len(all_positive)) / float(len(all_negative) + len(all_positive))
    folds_rv = [[] for _ in range(folds)]
    folds_size = len(values) / folds
    folds_pos_size = pos_ratio * folds_size
    folds_neg_size = folds_size - folds_pos_size
    np.random.shuffle(all_positive)
    np.random.shuffle(all_negative)
    for k in range(folds):
        # Stratified sampling, maintain same pos-neg ratio in all folds
        pos_idx = int(k * folds_pos_size)
        neg_idx = int(k * (folds_size - folds_pos_size))
        folds_rv[k] += all_positive[pos_idx:min(int(pos_idx + folds_pos_size), len(all_positive))]
        folds_rv[k] += all_negative[neg_idx:min(int(neg_idx + folds_neg_size), len(all_negative))]
        np.random.shuffle(folds_rv[k])
    return folds_rv
