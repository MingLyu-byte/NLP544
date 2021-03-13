import os
import sys
import numpy as np


def create_dict(training_set_path):
    f = open(training_set_path, "r", encoding='UTF-8')
    lines = f.readlines()
    emission_dict = {}
    transition_dict = {}
    tag_dict = {}
    vocab = []

    for line in lines:
        word_seq = line.strip().split(" ")
        prev_tag = "q0"
        tag_dict[prev_tag] = tag_dict.get(prev_tag, 0) + 1
        for i in range(len(word_seq)):
            cur_word_tag = word_seq[i].split("/")
            cur_tag = cur_word_tag[-1]
            start_find_index = len(word_seq[i]) - len(cur_tag)
            cur_tag_index = word_seq[i].find(cur_tag,start_find_index)
            cur_word = str(word_seq[i][:cur_tag_index - 1])

            transition_dict[(prev_tag, cur_tag)] = transition_dict.get((prev_tag, cur_tag), 0) + 1
            emission_dict[(cur_tag, cur_word)] = emission_dict.get((cur_tag, cur_word), 0) + 1
            tag_dict[cur_tag] = tag_dict.get(cur_tag, 0) + 1
            if cur_word not in vocab:
                vocab.append(cur_word)
            prev_tag = cur_tag

        transition_dict[(prev_tag, "qN")] = transition_dict.get((prev_tag, "qN"), 0) + 1
        tag_dict["qN"] = tag_dict.get("qN", 0) + 1

    return transition_dict, emission_dict, tag_dict, vocab


def create_transition_matrix(transition_dict, tag_dict, alpha=1):
    tag_seq = sorted(tag_dict.keys())
    num_tag = len(tag_dict)

    output = np.zeros((num_tag, num_tag))

    for i in range(num_tag):
        total = tag_dict[tag_seq[i]]
        for j in range(num_tag):
            transition_pair = (tag_seq[i], tag_seq[j])
            transition_count = transition_dict.get(transition_pair, 0)
            transition_p = (transition_count + alpha) / (total + alpha * num_tag)
            output[i, j] = transition_p

    return output


def create_emission_matrix(emission_dict, tag_dict, vocab):
    tag_seq = sorted(tag_dict.keys())
    num_words = len(vocab)
    num_tag = len(tag_dict)

    output = np.zeros((num_tag, num_words))

    for i in range(num_tag):
        total = tag_dict[tag_seq[i]]
        for j in range(num_words):
            emission_pair = (tag_seq[i], vocab[j])
            emission_count = emission_dict.get(emission_pair, 0)
            emission_p = float(emission_count) / float(total)
            output[i, j] = emission_p

    return output


def write_out_model(transition_matrix, emission_matrix, vocab, tag_dict):
    tag_seq = sorted(tag_dict.keys())
    f = open("hmmmodel.txt", "w", encoding='UTF-8')
    f.write(" ".join(vocab) + "\n")
    f.write(" ".join(tag_seq) + "\n")
    transition_matrix = transition_matrix.tolist()
    emission_matrix = emission_matrix.tolist()
    for i in range(len(tag_seq)):
        f.write(",".join([str(k) for k in transition_matrix[i]]) + "\n")

    for i in range(len(tag_seq)):
        f.write(",".join([str(k) for k in emission_matrix[i]]) + "\n")
    f.close()


if __name__ == '__main__':
    input = sys.argv[1]
    transition_dict, emission_dict, tag_dict, vocab = create_dict(input)
    transition_matrix = create_transition_matrix(transition_dict, tag_dict, 1)
    print(transition_matrix.shape)
    emission_matrix = create_emission_matrix(emission_dict, tag_dict, vocab)
    print(emission_matrix.shape)
    write_out_model(transition_matrix, emission_matrix, vocab, tag_dict)
