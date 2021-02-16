import os
import sys
import string

# Global Var
lable_class = [("negative", "deceptive"), ("negative", "truthful"), ("positive", "deceptive"), ("positive", "truthful")]
stop_words = ["the", 'a', "to", "and", "or", " "]


# read the model first line is log prior, the rest are log prior likelihood for each word in the training set
def read_model(path):
    f = open(path, 'r')
    log_prior = {}
    log_likelihood = {}

    log_prior_line = f.readline().strip().split(",")
    for i in range(len(lable_class)):
        log_prior[lable_class[i]] = float(log_prior_line[i])

    while True:
        line = f.readline()
        if line:
            log_likelihood_line = line.strip().split(",")
            word = log_likelihood_line[0]
            log_likelihood[word] = {}
            for i in range(1, len(log_likelihood_line)):
                log_likelihood[word][lable_class[i - 1]] = float(log_likelihood_line[i])
        else:
            break

    return log_prior, log_likelihood


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


# classify documents and write to output file
def classify_single(output, path, log_prior, log_likelihood):
    p = {}

    for key in log_prior.keys():
        p[key] = log_prior[key]

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            file_path = os.path.join(path, file)
            f = open(file_path)
            content = f.readlines()
            for line in content:
                review = process_reivew(line)
                for word in review:
                    if word in log_likelihood:
                        word_di = log_likelihood[word]
                        for key in p:
                            p[key] += word_di[key]

            predict_label = max(p, key=lambda key: p[key])
            output_line = predict_label[1] + " " + predict_label[0] + " " + file_path + "\n"
            output.write(output_line)

    f.close()


def classify_all(input, log_prior, log_likelihood):
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
                            classify_single(output, sub_fold_dir_path, log_prior, log_likelihood)


if __name__ == '__main__':
    input = sys.argv[1]
    log_prior, log_likelihood = read_model("nbmodel.txt")
    classify_all(input, log_prior, log_likelihood)
