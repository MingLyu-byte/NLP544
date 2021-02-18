import os
import sys
import string
from sklearn.metrics import f1_score
from nltk.corpus import stopwords

# Global Var
lable_class = [("negative", "deceptive"), ("negative", "truthful"), ("positive", "deceptive"), ("positive", "truthful")]
lable_class2 = ["positive", "negative", "truthful", "deceptive"]
# stop_words = ["the", 'a', "to", "and", "or", "between", "an", "both", "but"]
stop_words = list(stopwords.words('english'))


def read_model(path,lable_class):
    """
    read the model first line is log prior, the rest are log prior likelihood for each word in the training set
    :param path: string, input file path
    :param lable_class: list, lable_class for the model
    :return: dict, log_prior,
             dict, log_likelihood
    """
    f = open(path, 'r')
    log_prior = {}
    log_likelihood = {}

    # read log prior
    log_prior_line = f.readline().strip().split(",")
    for i in range(len(lable_class)):
        log_prior[lable_class[i]] = float(log_prior_line[i])

    while True:
        line = f.readline()
        if line:
            # read log likelihood
            log_likelihood_line = line.strip().split(",")
            word = log_likelihood_line[0]
            log_likelihood[word] = {}
            for i in range(1, len(log_likelihood_line)):
                log_likelihood[word][lable_class[i - 1]] = float(log_likelihood_line[i])
        else:
            break

    return log_prior, log_likelihood


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


def classify_single(output, path, log_prior, log_likelihood,lable_class):
    """
    classify documents and write to output file
    :param output: string, output file
    :param path: string, input directory path
    :param logprior: dict, trained log prior for each class
    :param loglikelihood: dict, trained log likelihood for each word
    :param lable_class: list, list, lable_class for the model
    :return: None
    """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            p = {}
            for key in log_prior.keys():
                p[key] = log_prior[key]
            file_path = os.path.join(path, file)
            f = open(file_path)
            content = f.readlines()
            for line in content:
                review = process_reivew(line)
                for word in review:
                    if word in log_likelihood:
                        word_di = log_likelihood[word]
                        for key in p.keys():
                            p[key] += word_di[key]

            if "positive" in lable_class:
                if p["positive"] > p["negative"]:
                    label2 = "positive"
                else:
                    label2 = "negative"

                if p["deceptive"] > p["truthful"]:
                    label1 = "deceptive"
                else:
                    label1 = "truthful"

                output_line = label1 + " " + label2 + " " + file_path + "\n"

            else:
                predict_label = max(p, key=lambda key: p[key])
                output_line = predict_label[1] + " " + predict_label[0] + " " + file_path + "\n"

            output.write(output_line)

    f.close()


def classify_all(input, log_prior, log_likelihood, label_class):
    """
    classify all the files
    :param input: string, input directory path
    :param logprior: dict, trained log prior for each class
    :param loglikelihood: dict, trained log likelihood for each word
    :param lable_class: list, list, lable_class for the model
    :return: None
    """
    output = open("nboutput.txt", "w")
    for dir in os.listdir(input):
        if not os.path.isfile(os.path.join(input, dir)):
            sub_directories = os.path.join(input, dir)
            for sub_dir in os.listdir(sub_directories):
                if not os.path.isfile(os.path.join(sub_directories, sub_dir)):
                    sub_folder_directories = os.path.join(sub_directories, sub_dir)
                    for sub_fold_dir in os.listdir(sub_folder_directories):
                        if not os.path.isfile(os.path.join(sub_folder_directories, sub_fold_dir)):
                            sub_fold_dir_path = os.path.join(sub_folder_directories, sub_fold_dir)
                            classify_single(output, sub_fold_dir_path, log_prior, log_likelihood,label_class)


if __name__ == '__main__':
    input_data = sys.argv[1]
    log_prior, log_likelihood = read_model("nbmodel.txt",lable_class)
    classify_all(input_data, log_prior, log_likelihood, lable_class)
