import sys
import os
import math
import string
from sklearn.metrics import f1_score
import numpy as np
import random

# Global Var
stop_words = ["the", 'a', "to", "and", "or", "between", "an", "both", "but"]
vocab_dict = {}
input_matrix = []
label_pn, label_dt = [], []
lable_class_dict = {"positive": 1, "negative": -1, "truthful": 1, "deceptive": -1}
answer = []
predict = []


def read_file_train(input_path, dict):
    for dir in os.listdir(input_path):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    count_review(4, sub_folder_directories, dict)


def generate_input_matrix(word_list, input_matrix, input_path, label_pn, label_dt):
    for dir in os.listdir(input_path):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            label_name_1 = dir.split("_")[0]
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    label_name_2 = sub_dir.split("_")[0]
                    generate_input_from_review(4, sub_folder_directories, word_list, label_name_1, label_name_2,
                                               label_pn, label_dt, input_matrix)


def generate_input_from_review(num, sub_folder_directories, word_list, label_name_1, label_name_2,
                               label_pn, label_dt, input_matrix):
    for i in range(1, num):
        fold = "fold" + str(i + 1)
        fold_path = os.path.join(sub_folder_directories, fold)
        for review_file in os.listdir(fold_path):
            label_pn.append(lable_class_dict[label_name_1])
            label_dt.append(lable_class_dict[label_name_2])
            with open(os.path.join(fold_path, review_file)) as f:
                content = f.readlines()
                for line in content:
                    words = process_reivew(line)
                    transform_review_as_feature(words, word_list, input_matrix)


def transform_review_as_feature(word_list, word_feature_list, input_matrix):
    temp = [0 for i in range(1000)]
    for word in word_list:
        if word in word_feature_list:
            index = word_feature_list.index(word)
            temp[index] = 1
    input_matrix.append(temp)


# count words from reviews, weird method to read only folds 2,3,4
def count_review(num, path, dict):
    for i in range(1, num):
        fold = "fold" + str(i + 1)
        fold_path = os.path.join(path, fold)
        for review_file in os.listdir(fold_path):
            with open(os.path.join(fold_path, review_file)) as f:
                content = f.readlines()
                for line in content:
                    word_list = process_reivew(line)
                    add_word(word_list, dict)


# helper Function
# add word count and extract the most frequent 1000 words as features later
def add_word(word_list, dict):
    for word in word_list:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1


# Extract the most frequent 1000 words as features
def extract_feature_words(dict):
    return sorted(dict, key=lambda k: dict[k], reverse=True)[:1000]


# Tokenize the sentence into word list "I love you" -> ["I","love","you"]
def process_reivew(review):
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
        temp = [0 for i in range(1000)]
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
        if pn == label1 and dt == label2:
            correct += 1

    print(label1, label2, ":", float(correct) / float(count))


def write_out(W_pn, b_pn, W_dt, b_dt, name, word_list):
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
    W_v_pn, b_v_pn = vanilla_perceptron_train(np.array(input_matrix), np.array(label_pn), 300)
    W_v_dt, b_v_dt = vanilla_perceptron_train(np.array(input_matrix), np.array(label_dt), 300)
    W_a_pn, b_a_pn = average_perceptron_train(np.array(input_matrix), np.array(label_pn), 300)
    W_a_dt, b_a_dt = average_perceptron_train(np.array(input_matrix), np.array(label_dt), 300)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/positive_polarity/truthful_from_TripAdvisor/fold1", 1, 1, word_list)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/positive_polarity/deceptive_from_MTurk/fold1", 1, -1, word_list)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/negative_polarity/deceptive_from_MTurk/fold1", -1, -1, word_list)
    model_test(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/negative_polarity/truthful_from_Web/fold1", -1, 1, word_list)

    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/positive_polarity/truthful_from_TripAdvisor/fold1", 1, 1, word_list)
    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/positive_polarity/deceptive_from_MTurk/fold1", 1, -1, word_list)
    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/negative_polarity/deceptive_from_MTurk/fold1", -1, -1, word_list)
    model_test(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "C:/Users/lyum/PycharmProjects/NLP544/HW3/op_spam_training_data"
                                               "/negative_polarity/truthful_from_Web/fold1", -1, 1, word_list)

    write_out(W_v_pn, b_v_pn, W_v_dt, b_v_dt, "vanillamodel",word_list)
    write_out(W_a_pn, b_a_pn, W_a_dt, b_a_dt, "averagedmodel",word_list)
