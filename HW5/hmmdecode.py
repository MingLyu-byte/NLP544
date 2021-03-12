import sys
import numpy as np
import math
import time


def read_model():
    f = open("hmmmodel.txt", "r", encoding='UTF-8')
    lines = f.readlines()
    vocab = lines[0].strip().split(" ")
    tag_seq = lines[1].strip().split(" ")
    num_tag = len(tag_seq)
    transition_matrix = []
    emission_matrix = []

    for i in range(2, 2 + num_tag):
        transition_matrix.append([float(k) for k in lines[i].strip().split(",")])

    for i in range(2 + num_tag, 2 + num_tag * 2):
        emission_matrix.append([float(k) for k in lines[i].strip().split(",")])

    transition_matrix = np.array(transition_matrix)
    emission_matrix = np.array(emission_matrix)
    print("transition_matrix dimension:", transition_matrix.shape)
    print("emission_matrix dimension:", emission_matrix.shape)

    return transition_matrix, emission_matrix, tag_seq, vocab


def viterbi_forward(word_seq, transition_matrix, emission_matrix, tag_seq, vocab):
    num_tags = len(tag_seq)
    num_words = len(word_seq)
    probs = np.zeros((num_tags, num_words))
    paths = np.zeros((num_tags, num_words))

    start_index = tag_seq.index("q0")
    end_index = tag_seq.index("qN")

    for i in range(num_tags):
        transition_p = math.log(transition_matrix[start_index, i])
        word = word_seq[0]
        if word in vocab:
            emission_p = emission_matrix[i, vocab.index(word)]
            if emission_p != 0:
                emission_p = math.log(emission_p)
                probs[i, 0] = transition_p + emission_p
            else:
                probs[i, 0] = 0
        else:
            probs[i, 0] = transition_p

    for i in range(1, num_words):
        word = word_seq[i]

        for j in range(num_tags):
            best_prob = float("-inf")
            best_path = None

            if word in vocab:
                emission_p = emission_matrix[j, vocab.index(word)]
                if emission_p == 0:
                    best_prob = float("-inf")
                    best_path = None
                else:
                    emission_p = math.log(emission_p)
                    for k in range(num_tags):
                        if probs[k, i - 1] == 0:
                            continue
                        else:
                            prob = probs[k, i - 1] + math.log(transition_matrix[k, j]) + emission_p
                            if prob > best_prob:
                                best_prob = prob
                                best_path = k
            else:
                for k in range(num_tags):
                    prob = probs[k, i - 1] + math.log(transition_matrix[k, j])
                    if prob > best_prob:
                        best_prob = prob
                        best_path = k

            probs[j, i] = best_prob
            paths[j, i] = best_path

    for i in range(num_tags):
        transition_p = math.log(transition_matrix[i, end_index])
        if probs[i, -1] != 0:
            probs[i, -1] += transition_p

    return probs, paths


def viterbi_backward(word_seq, tag_seq, probs, paths):
    num_tags = len(tag_seq)
    num_words = len(word_seq)
    tag_index_seq = [None] * num_words
    pred_tag = [None] * num_words
    best_prob = float("-inf")

    for i in range(num_tags):
        if probs[i, -1] == 0:
            continue

        if probs[i, -1] > best_prob:
            best_prob = probs[i, -1]
            tag_index_seq[-1] = i

    pred_tag[-1] = tag_seq[tag_index_seq[-1]]

    for i in range(num_words - 1, 0, -1):
        tag_index_seq[i - 1] = paths[int(tag_index_seq[i]), i]
        pred_tag[i - 1] = tag_seq[int(tag_index_seq[i - 1])]

    return pred_tag


def predict(file_path, transition_matrix, emission_matrix, tag_seq, vocab):
    f = open(file_path, "r", encoding='UTF-8')
    output = open("hmmoutput.txt", "w", encoding='UTF-8')
    lines = f.readlines()
    f.close()
    count = 0
    for line in lines:
        word_seq = line.strip().split(" ")
        probs, paths = viterbi_forward(word_seq, transition_matrix, emission_matrix, tag_seq, vocab)
        pred_tag = viterbi_backward(word_seq, tag_seq, probs, paths)
        temp = ""
        for i in range(len(word_seq)):
            if i == len(word_seq) - 1:
                temp = temp + word_seq[i] + "/" + pred_tag[i] + "\n"
            else:
                temp = temp + word_seq[i] + "/" + pred_tag[i] + " "

        output.write(temp)
    output.close()


if __name__ == '__main__':
    input = sys.argv[1]
    transition_matrix, emission_matrix, tag_seq, vocab = read_model()
    # probs, paths = viterbi_forward( ["Ecco", "l", "arringa", "di", "Tiziana", "Maiolo", "."], transition_matrix,
    #                                emission_matrix, tag_seq, vocab)
    # pred_tag = viterbi_backward( ["Ecco", "l", "arringa", "di", "Tiziana", "Maiolo", "."], tag_seq, probs, paths)
    start = time.time()
    predict(input, transition_matrix, emission_matrix, tag_seq, vocab)
    end = time.time()
    print(end - start)
