import sys
import os
import math
import string
from sklearn.metrics import f1_score
from nltk.corpus import stopwords

# Global Var
# stop_words = ["the", 'a', "to", "and", "or", "between", "an", "both", "but"]
stop_words = list(stopwords.words('english'))
vocab_dict = {}
label_count = {}
lable_class = [("negative", "deceptive"), ("negative", "truthful"), ("positive", "deceptive"), ("positive", "truthful")]
lable_class_dict = {("negative", "deceptive"): 1, ("negative", "truthful"): 2, ("positive", "deceptive"): 3,
                    ("positive", "truthful"): 4}
lable_class2 = ["positive", "negative", "truthful", "deceptive"]


def read_file_train(input, dict):
    """
    read training files from the input path and store them into the dictionary, read only folds 2,3,4
    :param input: string, input directory path, 3 directories deep
    :param dict: dict, dictionary to store the data
    :return: None
    """
    for dir in os.listdir(input):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            # extract label1: positive or negative
            label_name_1 = dir.split("_")[0]
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    # extract label2: truthful or deceptive
                    label_name_2 = sub_dir.split("_")[0]
                    count_review_train(4, sub_folder_directories, label_name_1, label_name_2, dict)


def count_review_train(num, path, label1, label2, dict):
    """
    count number of times each word appear in each class for further use
    :param num: int, fold number to read from
    :param path: string, input directory path
    :param label1: string, positive or negative
    :param label2: string, truthful or deceptive
    :param dict: dict, dictionary to store the data
    :return: None
    """
    for i in range(1, num):
        # read fold 2,3,4
        fold = "fold" + str(i + 1)
        fold_path = os.path.join(path, fold)
        for review_file in os.listdir(fold_path):
            with open(os.path.join(fold_path, review_file)) as f:
                content = f.readlines()
                # store lable count
                label_count[(label1, label2)] = label_count.get((label1, label2), 0) + 1
                for line in content:
                    # Tokenize the review sentence
                    word_list = process_reivew(line)
                    # Store the word count into dictionary
                    add_word(word_list, label1, label2, dict)


def read_file_all(input, dict):
    """
    read training files from the input path and store them into the dictionary
    :param input: string, input directory path, 3 directories deep
    :param dict: dict, dictionary to store the data
    :return: None
    """
    for dir in os.listdir(input):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            label_name_1 = dir.split("_")[0]
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    label_name_2 = sub_dir.split("_")[0]
                    for sub_fold_dir in os.listdir(sub_folder_directories):
                        if not os.path.isfile(os.path.join(sub_folder_directories, sub_fold_dir)):
                            count_all_review(os.path.join(sub_folder_directories, sub_fold_dir),
                                             label_name_1, label_name_2, dict)


def count_all_review(path, label1, label2, dict):
    """
    count number of times each word appear in each class for further use
    :param path: string, input directory path
    :param label1: string, positive or negative
    :param label2: string, truthful or deceptive
    :param dict: dict, dictionary to store the data
    :return: None
    """
    for dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, dir)):
            fold_path = os.path.join(path, dir)
            f = open(fold_path, "r")
            content = f.readlines()
            label_count[(label1, label2)] = label_count.get((label1, label2), 0) + 1
            for line in content:
                # Tokenize the review sentence
                word_list = process_reivew(line)
                # Store the word count into dictionary
                add_word(word_list, label1, label2, dict)


def train_naive_bayes_2_class(dict):
    """
    naive bayes 2 classes classifier using the word count dict
    :param dict: word count dictionary
    :return: dict, log_prior,
             dict, log_likelihood
    """
    N_nd = N_nt = N_pd = N_pt = 0
    log_likelihood = {}
    log_prior = {}

    # count number of unique words for smoothing later
    vocab = set([pair[0] for pair in dict.keys()])
    V = len(vocab)

    # log prior
    count_sum = sum(label_count.values())
    log_prior['positive'] = math.log2(
        label_count[("positive", "truthful")] + label_count[("positive", "deceptive")]) - math.log2(count_sum)
    log_prior['negative'] = math.log2(
        label_count[("negative", "truthful")] + label_count[("negative", "deceptive")]) - math.log2(count_sum)
    log_prior['deceptive'] = math.log2(
        label_count[("positive", "deceptive")] + label_count[("negative", "deceptive")]) - math.log2(count_sum)
    log_prior['truthful'] = math.log2(
        label_count[("negative", "truthful")] + label_count[("positive", "truthful")]) - math.log2(count_sum)

    # log likelihood total count
    for pair in dict:
        if pair[1] == "negative":
            if pair[2] == "deceptive":
                N_nd += dict[pair]
            else:
                N_nt += dict[pair]
        else:
            if pair[2] == "deceptive":
                N_pd += dict[pair]
            else:
                N_pt += dict[pair]

    # log likelihood probability for each word
    for word in vocab:
        freq_nd = dict.get((word, "negative", "deceptive"), 0)
        freq_nt = dict.get((word, "negative", "truthful"), 0)
        freq_pd = dict.get((word, "positive", "deceptive"), 0)
        freq_pt = dict.get((word, "positive", "truthful"), 0)

        p_n = (freq_nd + freq_nt + 1) / (N_nd + N_nt + V)
        p_p = (freq_pd + freq_pt + 1) / (N_pd + N_pt + V)
        p_d = (freq_pd + freq_nd + 1) / (N_pd + N_nd + V)
        p_t = (freq_pt + freq_nt + 1) / (N_pt + N_nt + V)

        log_likelihood[word] = {"negative": math.log2(p_n), "truthful": math.log2(p_t),
                                "positive": math.log2(p_p), "deceptive": math.log2(p_d)}

    return log_prior, log_likelihood


def train_naive_bayes_4_class(dict):
    """
    naive bayes 4 classes classifier using the word count dict
    :param dict: word count dictionary
    :return: dict, log_prior,
             dict, log_likelihood
    """
    N_nd = N_nt = N_pd = N_pt = 0
    log_likelihood = {}
    log_prior = {}

    # count number of unique words for smoothing later
    vocab = set([pair[0] for pair in dict.keys()])
    V = len(vocab)

    # log prior
    count_sum = sum(label_count.values())
    for key in label_count.keys():
        temp_count = label_count[key]
        log_prior[key] = math.log2(temp_count) - math.log2(count_sum)

    # log likelihood total count
    for pair in dict:
        if pair[1] == "negative":
            if pair[2] == "deceptive":
                N_nd += dict[pair]
            else:
                N_nt += dict[pair]
        else:
            if pair[2] == "deceptive":
                N_pd += dict[pair]
            else:
                N_pt += dict[pair]

    # log likelihood probability for each word
    for word in vocab:
        freq_nd = dict.get((word, "negative", "deceptive"), 0)
        freq_nt = dict.get((word, "negative", "truthful"), 0)
        freq_pd = dict.get((word, "positive", "deceptive"), 0)
        freq_pt = dict.get((word, "positive", "truthful"), 0)

        p_nd = (freq_nd + 1) / (N_nd + 1 * V)
        p_nt = (freq_nt + 1) / (N_nt + 1 * V)
        p_pd = (freq_pd + 1) / (N_pd + 1 * V)
        p_pt = (freq_pt + 1) / (N_pt + 1 * V)

        log_likelihood[word] = {("negative", "deceptive"): math.log2(p_nd), ("negative", "truthful"): math.log2(p_nt),
                                ("positive", "deceptive"): math.log2(p_pd), ("positive", "truthful"): math.log2(p_pt)}

    return log_prior, log_likelihood


# Helper Function
def add_word(word_list, label1, label2, dict):
    """
    add word count in each class (how many times the word appear in documents of different classes)
    :param word_list: list, a list of words
    :param label1: string, positive or negative
    :param label2: string, truthful or deceptive
    :param dict: dict, dictionary to store the data
    :return: None
    """
    for word in word_list:
        word_pair = (word, label1, label2)
        dict[word_pair] = dict.get(word_pair, 0) + 1


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


def remove_freq(vocab_dict):
    """
    remove most frequent the most infrequent elements in vocab_dict
    :param vocab_dict:
    :return: dict, dictionary to store the data
    """
    most_common_list = sorted(vocab_dict.keys(), reverse=True)[:5]
    most_uncommon_list = sorted(vocab_dict.keys())[:5]
    for key in most_common_list:
        del vocab_dict[key]
    for key in most_uncommon_list:
        del vocab_dict[key]


def naive_bayes_predict(path, logprior, loglikelihood, correct_label, predicted, answer):
    """
    Test our naive bayes model
    :param path: string, input directory path
    :param logprior: dict, trained log prior for each class
    :param loglikelihood: dict, trained log likelihood for each word
    :param correct_label: tuple, correct label
    :param predicted: list, empty array to store predict label
    :param answer: list, empty array to store correct label
    :return: None
    """
    count = 0
    correct = 0

    for file in os.listdir(path):
        p = {}
        for key in logprior.keys():
            # initialize with log prior
            p[key] = logprior[key]
        count += 1
        f = open(os.path.join(path, file))
        content = f.readlines()
        for line in content:
            review = process_reivew(line)
            for word in review:
                if word in loglikelihood:
                    word_di = loglikelihood[word]
                    for key in p:
                        p[key] += word_di[key]

        # find label with maximum probability
        predict_label = max(p, key=lambda key: p[key])
        predicted.append(lable_class_dict[predict_label])
        answer.append(lable_class_dict[correct_label])
        if predict_label == correct_label:
            correct += 1
    print(correct_label, ":", float(correct) / float(count))


def write_out(log_prior, log_likelihood, lable_class):
    """
    write out method to store the model
    :param log_prior: dict, trained log prior for each class
    :param log_likelihood: dict, trained log likelihood for each word
    :param lable_class: list, list of lable classes
    :return: None
    """
    f = open("nbmodel.txt", "w")
    # write out prior in the same order as the label_class list
    lable_prior = [str(log_prior[lable_class[i]]) for i in range(len(lable_class))]
    output = ",".join(lable_prior) + "\n"
    f.write(output)

    # write out log likelihood for each word in the same order as the label_class list
    for word in log_likelihood.keys():
        word_likelihood = [str(log_likelihood[word][lable_class[i]]) for i in range(len(lable_class))]
        output = ",".join(word_likelihood)
        output = word + "," + output + "\n"
        f.write(output)

    f.close()


if __name__ == '__main__':
    input = sys.argv[1]
    read_file_train(input, vocab_dict)
    remove_freq(vocab_dict)
    log_prior, log_likelihood = train_naive_bayes_4_class(vocab_dict)
    log_prior2, log_likelihood2 = train_naive_bayes_2_class(vocab_dict)
    predicted = []
    answer = []
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/positive_polarity/truthful_from_TripAdvisor/fold1", log_prior, log_likelihood,
                        ("positive", "truthful"), predicted, answer)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/positive_polarity/deceptive_from_MTurk/fold1", log_prior, log_likelihood,
                        ("positive", "deceptive"), predicted, answer)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/negative_polarity/deceptive_from_MTurk/fold1", log_prior, log_likelihood,
                        ("negative", "deceptive"), predicted, answer)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/negative_polarity/truthful_from_Web/fold1", log_prior, log_likelihood,
                        ("negative", "truthful"), predicted, answer)
    write_out(log_prior, log_likelihood, lable_class)
    print("f1 score:", f1_score(answer, predicted, average="macro"))
