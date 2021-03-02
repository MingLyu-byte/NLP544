import sys
import os
import math
import string
from sklearn.metrics import f1_score
import numpy as np

# Global Var
stop_words = ["the", 'a', "to", "and", "or", "between", "an", "both", "but"]
vocab_dict = {}
input_matrix = []
label_pn, label_dt = [], []
lable_class_dict = {"positive": 1, "negative": -1, "truthful": 1, "deceptive": -1}
answer = []
predict = []


def read_file_train(input_path, dict):
    """
    read files to train on folds 2,3,4
    :param input_path: string, path to input folder
    :param dict: dict, dictionary to contain words info (word,count)
    :return: none
    """
    for dir in os.listdir(input_path):
        if not os.path.isfile(os.path.join(input_path, dir)):
            sub_directories = os.path.join(input_path, dir)
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    count_review_train(4, sub_folder_directories, dict)


def count_review_train(num, path, dict):
    """
    count words from reviews, weird method to read only folds 2,3,4
    :param num: int, number to extract folders
    :param path: string, input path
    :param dict: dict, dictionary to contain words info (word,count)
    :return: none
    """
    for i in range(1, num):
        fold = "fold" + str(i + 1)
        fold_path = os.path.join(path, fold)
        for review_file in os.listdir(fold_path):
            with open(os.path.join(fold_path, review_file)) as f:
                content = f.readlines()
                for line in content:
                    word_list = process_reivew(line)
                    add_word(word_list, dict)


def read_file_all(input_path, dict):
    """
    read files to train
    :param input_path: string, path to input folder
    :param dict: dict, dictionary to contain words info (word,count)
    :return: none
    """
    for dir in os.listdir(input_path):
        if not os.path.isfile(os.path.join(input_path, dir)):
            sub_directories = os.path.join(input_path, dir)
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    count_review(sub_folder_directories, dict)


def count_review(path, dict):
    """
    count words from reviews
    :param path: string, input path
    :param dict: dict, dictionary to contain words info (word,count)
    :return: none
    """
    for dir in os.listdir(path):
        fold_path = os.path.join(path, dir)
        for review_file in os.listdir(fold_path):
            if os.path.isfile(os.path.join(fold_path, review_file)):
                with open(os.path.join(fold_path, review_file)) as f:
                    content = f.readlines()
                    for line in content:
                        word_list = process_reivew(line)
                        add_word(word_list, dict)


def generate_input_matrix(word_list, input_matrix, input_path, label_pn, label_dt):
    """
    generate input matrix to train.
    :param word_list: list, word list to train on
    :param input_matrix: 2d array, each row is a data point (train file), each column is a feature (word from word list)
    :param input_path: string, input directory path
    :param label_pn: int, label for positive (1) or negative (-1)
    :param label_dt: int, label for deceptive (-1) or truthful (1)
    :return: None
    """
    for dir in os.listdir(input_path):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            label_name_1 = dir.split("_")[0]
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    label_name_2 = sub_dir.split("_")[0]
                    generate_input_from_review(sub_folder_directories, word_list, label_name_1, label_name_2,
                                               label_pn, label_dt, input_matrix)


def generate_input_from_review(sub_folder_directories, word_list, label_name_1, label_name_2,
                               label_pn, label_dt, input_matrix):
    """
    extract information from review to generate each data point.
    :param sub_folder_directories: string, directory path
    :param word_list: list, word list to train on
    :param label_name_1: int, correct lable for positive (1) or negative (-1)
    :param label_name_2: int, correct label for deceptive (-1) or truthful (1)
    :param label_pn: string, label "positive" or "negative"
    :param label_dt: string, label "deceptive" or "truthful"
    :param input_matrix: 2d array, each row is a data point (train file),
    each column is a feature (word from word list)
    :return: None
    """
    for dir in os.listdir(sub_folder_directories):
        fold_path = os.path.join(sub_folder_directories, dir)
        for review_file in os.listdir(fold_path):
            if os.path.isfile(os.path.join(fold_path, review_file)):
                label_pn.append(lable_class_dict[label_name_1])
                label_dt.append(lable_class_dict[label_name_2])
                with open(os.path.join(fold_path, review_file)) as f:
                    content = f.readlines()
                    for line in content:
                        words = process_reivew(line)
                        transform_review_as_feature(words, word_list, input_matrix)


def transform_review_as_feature(word_list, word_feature_list, input_matrix):
    """
    helper method to generate each data point, if the word in word list appears at least once,
    we set the value at the index of word appearance to 1. Otherwise, we leave it 0.
    :param word_list: list, tokenized review as a list to train on
    :param word_feature_list: list, word list to train on
    :param input_matrix: 2d array, each row is a data point (train file),
    each column is a feature (word from word list)
    :return: None
    """
    temp = [0 for i in range(1500)]
    for word in word_list:
        if word in word_feature_list:
            index = word_feature_list.index(word)
            temp[index] = 1
    input_matrix.append(temp)


# helper Function
# add word count and extract the most frequent 1000 words as features later
def add_word(word_list, dict):
    """
    add word count and extract the most frequent 1500 words as features later
    :param word_list: list, tokenized review as a list to train on
    :param dict: dict, dictionary to contain words info (word,count)
    :return: None
    """
    for word in word_list:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1


def extract_feature_words(dict):
    """
    Extract the most frequent 1500 words as features
    :param dict: dict, dictionary to contain words info (word,count)
    :return: None
    """
    return sorted(dict, key=lambda k: dict[k], reverse=True)[:1500]


def process_reivew(review):
    """
    Tokenize the sentence into word list "I love you" -> ["I","love","you"]
    :param review: string, a review sentence
    :return: review_clean: string, tokenized review as a list of words
    """
    review_split = review.strip().split(" ")
    # remove puncuation
    review_nopunc = [s.translate(str.maketrans('', '', string.punctuation)) for s in review_split]
    # remove none value
    review_nonone = [i for i in review_nopunc if i]
    # lower case all words
    review_lowercase = [i.lower() for i in review_nonone]
    # remove stop words
    review_clean = [item for item in review_lowercase if item not in stop_words]
    return review_clean


def vanilla_perceptron_train(input, label, maxiter=50):
    """
    Vanilla perceptron algorithm
    :param input: 2d array, input matrix
    :param label: 1d array, label vector
    :param maxiter: int, number of iterations to do
    :return: W, 1d array
             b, int
    """
    W = np.zeros(input.shape[1])
    b = 0
    a = 0
    for i in range(maxiter):
        for x, y in zip(input, label):
            a += np.dot(W, x) + b
            if y * a <= 0:
                W += y * x
                b += y
            a = 0

    return W, b


def average_perceptron_train(input, label, maxiter=50):
    """
    average perceptron algorithm
    :param input: 2d array, input matrix
    :param label: 1d array, label vector
    :param maxiter: int, number of iterations to do
    :return: W, 1d array
             b, int
    """
    W = np.zeros(input.shape[1])
    U = np.zeros(input.shape[1])
    b = 0
    beta = 0
    c = 1
    for i in range(maxiter):
        for x, y in zip(input, label):
            a = 0
            a += np.dot(W, x) + b
            if y * a <= 0:
                W += y * x
                b += y
                U += y * c * x
                beta += y * c

            c += 1
    return W - 1 / c * U, b - 1 / c * beta


def model_test(W_pn, b_pn, W_dt, b_dt, input_path, label1, label2, word_feature_list):
    count = 0
    correct = 0
    for file in os.listdir(input_path):
        temp = [0 for i in range(1500)]
        count += 1
        f = open(os.path.join(input_path, file))
        content = f.readlines()
        for line in content:
            review = process_reivew(line)
            for word in review:
                if word in word_feature_list:
                    index = word_feature_list.index(word)
                    temp[index] = 1
        pn = np.sign(np.dot(W_pn, np.array(temp)) + b_pn)
        dt = np.sign(np.dot(W_dt, np.array(temp)) + b_dt)
        if np.abs(pn - label1) == 0 and np.abs(dt - label2) == 0:
            correct += 1

    print(label1, label2, ":", float(correct) / float(count))


def write_out(W_pn, b_pn, W_dt, b_dt, name, word_list):
    """
    write out method to write out weights, bias, word feature list and name of the model
    :param W_pn: weights for positive and negative labels
    :param b_pn: bias for positive and negative labels
    :param W_dt: weights for deceptive and truthful labels
    :param b_dt: bias for deceptive and truthful labels
    :param name: name of the model
    :param word_list: list, word list to train on
    :return:
    """
    file_name = name + ".txt"
    f = open(file_name, "w")
    f.write(",".join(word_list) + "\n")
    W_pn = [str(i) for i in list(W_pn)]
    W_dt = [str(i) for i in list(W_dt)]
    f.write(",".join(W_pn) + "\n")
    f.write(str(b_pn) + "\n")
    f.write(",".join(W_dt) + "\n")
    f.write(str(b_dt))
    f.close()


if __name__ == '__main__':
    input = sys.argv[1]
    read_file_train(input, vocab_dict)
    word_list = extract_feature_words(vocab_dict)
    generate_input_matrix(word_list, input_matrix, input, label_pn, label_dt)
    # shuffle the input matrix and label vector to mix up the order to get more robust result
    input_matrix = np.array(input_matrix)
    label_dt = np.array(label_dt)
    label_pn = np.array(label_pn)
    s = np.arange(0, len(label_dt), 1)
    np.random.shuffle(s)
    input_matrix = input_matrix[s]
    label_dt = label_dt[s]
    label_pn = label_pn[s]
    W_v_pn, b_v_pn = vanilla_perceptron_train(input_matrix, label_pn, 200)
    W_v_dt, b_v_dt = vanilla_perceptron_train(input_matrix, label_dt, 200)
    W_a_pn, b_a_pn = average_perceptron_train(input_matrix, label_pn, 200)
    W_a_dt, b_a_dt = average_perceptron_train(input_matrix, label_dt, 200)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "op_spam_training_data"
                                               "/positive_polarity/truthful_from_TripAdvisor/fold1", 1, 1, word_list)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "op_spam_training_data"
                                               "/positive_polarity/deceptive_from_MTurk/fold1", 1, -1, word_list)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "op_spam_training_data"
                                               "/negative_polarity/deceptive_from_MTurk/fold1", -1, -1, word_list)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "op_spam_training_data"
                                               "/negative_polarity/truthful_from_Web/fold1", -1, 1, word_list)

    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "op_spam_training_data"
                                               "/positive_polarity/truthful_from_TripAdvisor/fold1", 1, 1, word_list)
    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "op_spam_training_data"
                                               "/positive_polarity/deceptive_from_MTurk/fold1", 1, -1, word_list)
    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "op_spam_training_data"
                                               "/negative_polarity/deceptive_from_MTurk/fold1", -1, -1, word_list)
    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "op_spam_training_data"
                                               "/negative_polarity/truthful_from_Web/fold1", -1, 1, word_list)

    write_out(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "vanillamodel", word_list)
    write_out(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "averagedmodel", word_list)
