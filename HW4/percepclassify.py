import os
import sys
import os
import sys
import numpy as np
import string

# Global Var
stop_words = ["the", 'a', "to", "and", "or", "between", "an", "both", "but"]


def read_model(path):
    """
    read the model from the model path
    :param path: string, path to the model file
    :return: word_list: list, list of words to train on
             W_pn: weights for positive and negative labels
             b_pn: bias for positive and negative labels
             W_dt: weights for deceptive and truthful labels
             b_dt: bias for deceptive and truthful labels
    """
    f = open(path, "r")
    word_list = f.readline().strip().split(",")
    W_pn = [float(i) for i in f.readline().strip().split(",")]
    b_pn = float(f.readline().strip())
    W_dt = [float(i) for i in f.readline().strip().split(",")]
    b_dt = float(f.readline().strip())

    return word_list,W_pn, b_pn, W_dt, b_dt


def classify_all(input, W_pn, b_pn, W_dt, b_dt, word_feature_list):
    """
    classify all files in the folder
    :param input: string, input path to the folder
    :param W_pn: weights for positive and negative labels
    :param b_pn: bias for positive and negative labels
    :param W_dt: weights for deceptive and truthful labels
    :param b_dt: bias for deceptive and truthful labels
    :param word_feature_list: list, list of words to train on
    :return: None
    """
    output = open("percepoutput.txt", "w")
    for dir in os.listdir(input):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    for sub_fold_dir in os.listdir(sub_folder_directories):
                        if not os.path.isfile(os.path.join(sub_folder_directories, sub_fold_dir)):
                            sub_fold_dir_path = os.path.join(sub_folder_directories, sub_fold_dir)
                            classify_single(output, sub_fold_dir_path, W_pn, b_pn, W_dt, b_dt, word_feature_list)


def classify_single(output, sub_fold_dir_path, W_pn, b_pn, W_dt, b_dt, word_feature_list):
    """
    classify all files in the folder, helper method
    :param output: string, output path to write the predict label
    :param sub_fold_dir_path: string, input path for the training folder path
    :param W_pn: weights for positive and negative labels
    :param b_pn: bias for positive and negative labels
    :param W_dt: weights for deceptive and truthful labels
    :param b_dt: bias for deceptive and truthful labels
    :param word_feature_list: list, list of words to train on
    :return: None
    """
    label1 = ""
    label2 = ""
    for file in os.listdir(sub_fold_dir_path):
        temp = [0 for i in range(1500)]
        f = open(os.path.join(sub_fold_dir_path, file))
        content = f.readlines()
        for line in content:
            review = process_reivew(line)
            for word in review:
                if word in word_feature_list:
                    index = word_feature_list.index(word)
                    temp[index] = 1
        pn = np.sign(np.dot(W_pn, np.array(temp)) + b_pn)
        dt = np.sign(np.dot(W_dt, np.array(temp)) + b_dt)

        if pn >= 0:
            label1 = "positive"
        else:
            label1 = "negative"

        if dt >= 0:
            label2 = "truthful"
        else:
            label2 = "deceptive"

        output.write(label2 + " " + label1 + " " + os.path.join(sub_fold_dir_path, file) + "\n")


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


if __name__ == '__main__':
    input = sys.argv[1]
    word_list, W_pn, b_pn, W_dt, b_dt = read_model("vanillamodel.txt")
    classify_all(input, W_pn, b_pn, W_dt, b_dt, word_list)
    print("End")
