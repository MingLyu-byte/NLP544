import sys
import os
import math
import string
from sklearn.metrics import f1_score

# Global Var
stop_words = ["the", 'a', "to", "and", "or", "between", "an", "both", "but"]
vocab_dict = {}
label_count = {}
lable_class = [("negative", "deceptive"), ("negative", "truthful"), ("positive", "deceptive"), ("positive", "truthful")]
lable_class_dict = {("negative", "deceptive"):1,("negative", "truthful"):2,("positive", "deceptive"):3,
                    ("positive", "truthful"):4}

def read_file_train(input, dict):
    for dir in os.listdir(input):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            label_name_1 = dir.split("_")[0]
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    label_name_2 = sub_dir.split("_")[0]
                    count_review(4, sub_folder_directories, label_name_1, label_name_2, dict)


def read_file_all(input, dict):
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


def train_naive_bayes(dict):
    N_nd = N_nt = N_pd = N_pt = 0
    log_likelihood = {}
    log_prior = {}
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

        p_nd = (freq_nd + 1) / (N_nd + V)
        p_nt = (freq_nt + 1) / (N_nt + V)
        p_pd = (freq_pd + 1) / (N_pd + V)
        p_pt = (freq_pt + 1) / (N_pt + V)

        log_likelihood[word] = {("negative", "deceptive"): math.log2(p_nd), ("negative", "truthful"): math.log2(p_nt),
                                ("positive", "deceptive"): math.log2(p_pd), ("positive", "truthful"): math.log2(p_pt)}

    return log_prior, log_likelihood


def naive_bayes_predict(path, logprior, loglikelihood, correct_label,predicted,answer):
    count = 0
    correct = 0
    p = {}
    for key in logprior.keys():
        p[key] = logprior[key]

    for file in os.listdir(path):
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

        predict_label = max(p, key=lambda key: p[key])
        predicted.append(lable_class_dict[predict_label])
        answer.append(lable_class_dict[correct_label])
        if predict_label == correct_label:
            correct += 1

    print(correct_label, ":", float(correct) / float(count))


# write out method to store the model
def write_out(log_prior, log_likelihood):
    f = open("nbmodel.txt", "w")
    lable_prior = [str(log_prior[lable_class[i]]) for i in range(len(lable_class))]
    output = ",".join(lable_prior) + "\n"
    f.write(output)

    for word in log_likelihood.keys():
        word_likelihood = [str(log_likelihood[word][lable_class[i]]) for i in range(len(lable_class))]
        output = ",".join(word_likelihood)
        output = word + "," + output + "\n"
        f.write(output)

    f.close()


def count_review(num, path, label1, label2, dict):
    for i in range(1, num):
        fold = "fold" + str(i + 1)
        fold_path = os.path.join(path, fold)
        for review_file in os.listdir(fold_path):
            with open(os.path.join(fold_path, review_file)) as f:
                content = f.readlines()
                label_count[(label1, label2)] = label_count.get((label1, label2), 0) + 1
                for line in content:
                    word_list = process_reivew(line)
                    add_word(word_list, label1, label2, dict)


def count_all_review(path, label1, label2, dict):
    for dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, dir)):
            fold_path = os.path.join(path, dir)
            f = open(fold_path,"r")
            content = f.readlines()
            label_count[(label1, label2)] = label_count.get((label1, label2), 0) + 1
            for line in content:
                word_list = process_reivew(line)
                add_word(word_list, label1, label2, dict)


# helper Function
# add word count in each class (how many times the word appear in documents of different classes)
def add_word(word_list, label1, label2, dict):
    for word in word_list:
        word_pair = (word, label1, label2)
        dict[word_pair] = dict.get(word_pair, 0) + 1


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


if __name__ == '__main__':
    input = sys.argv[1]
    predicted = []
    answer = []
    read_file_all(input, vocab_dict)
    log_prior, log_likelihood = train_naive_bayes(vocab_dict)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/positive_polarity/truthful_from_TripAdvisor/fold1", log_prior, log_likelihood,
                        ("positive", "truthful"),predicted,answer)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/positive_polarity/deceptive_from_MTurk/fold1", log_prior, log_likelihood,
                        ("positive", "deceptive"),predicted,answer)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/negative_polarity/deceptive_from_MTurk/fold1", log_prior, log_likelihood,
                        ("negative", "deceptive"),predicted,answer)
    naive_bayes_predict("C:/Users/Lyu/PycharmProjects/NLP544/HW3/op_spam_training_data"
                        "/negative_polarity/truthful_from_Web/fold1", log_prior, log_likelihood,
                        ("negative", "truthful"),predicted,answer)
    write_out(log_prior, log_likelihood)
    print("macro f1:",f1_score(answer, predicted, average='macro'))
