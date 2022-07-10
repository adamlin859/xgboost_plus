
import os
import io
import scipy.io as matio
import numpy as np
import re


INGREDIENT_PATH = "data/raw/food101/metadata/ingredients.txt"
INTERMIN_PATH = "data/interim/"

def loadMat(root_path, fileName):
    file = matio.loadmat(root_path + fileName)[fileName[:-4]]
    return file


def preprocess(opt):
    root_path = opt.data_path
    raw_ingre_info = os.path.join(root_path, INGREDIENT_PATH)

    # create interim data
    ingreList, ingredient_all_feature = parse_ingre_presence(raw_ingre_info, opt)
    
    # create LSTM input features
    indexVector = create_LSTM_input(ingredient_all_feature, opt)

    #get ingredient words
    ingre2word_map, wordList = get_ingre_term2word_map(ingreList, opt)

    #get embedding of 461 ingredients
    wordVector_word = create_glove_matrix(wordList, opt)

    #get embeddings of 446 ingredient terms
    wordVector = np.zeros([446,300])

    #process each ingredient term
    for i in range(wordVector.shape[0]):
        #get the ingre words for the i-th ingredient term
        ingre_word_indicator = ingre2word_map[i]
        index_words = np.where(ingre_word_indicator>0)[0]

        #get ingre term embedding by sum of those of words
        wordVector[i] += wordVector_word[index_words].sum(0)

    matio.savemat(opt.data_path + INTERMIN_PATH + 'wordVector.mat', {'wordVector': wordVector})
    #-------------------------------------------------------------------------

def parse_ingre_presence(raw_ingre_info, opt):
    # read ingredient data
    with io.open(raw_ingre_info, encoding='utf-8') as file:
        lines = file.read().split('\n')

    # construct ingre dict and the multi-hot ingredient distributions of all 101 classes
    num_class = len(lines)
    max_num_word = 500

    ingreList = []
    ingredient_all_feature = np.zeros((num_class, max_num_word))

    #process each line of ingredients
    for i in range(num_class):

        #get a line of ingredients
        line = lines[i]
        ingredients = line.split(',')  # line is a string

        #check each of the ingredient
        num_ingre = len(ingredients)
        for j in range(num_ingre):
            ingredient = ingredients[j]
            if ingredient in ingreList: #fill the feature vector if ingredient in our dict
                ingre_index = ingreList.index(ingredient)
                ingredient_all_feature[i,ingre_index] = 1
            else: # expand dict and fill feature vector
                ingreList.append(ingredient)
                ingredient_all_feature[i, len(ingreList)-1] = 1


    matio.savemat(opt.data_path + INTERMIN_PATH + 'ingreList.mat', {'ingreList': ingreList})
    matio.savemat(opt.data_path + INTERMIN_PATH + 
        'ingredient_all_feature.mat', {'ingredient_all_feature': ingredient_all_feature})

    return ingreList, ingredient_all_feature


def create_LSTM_input(ingredient_all_feature, opt):
    max_seq = 30
    num_data = len(ingredient_all_feature)

    # construct indexVectors
    indexVector = np.zeros((num_data, max_seq))  # store the input seq of each class
    seq_max = 0
    seq_avg = 0

    for i in range(0, num_data):  # for ingre vector of each class
        # print('processing data ' + str(i))

        # get the indexes of ingredient terms
        data = ingredient_all_feature[i]
        index_term = np.where(data > 0)[0]        

        # fill indexVector
        len_seq = len(index_term)
        indexVector[i, :len_seq] += index_term + 1 #get 1-indexed ingredients

        if len_seq > seq_max:
            seq_max = len_seq
        seq_avg += len_seq

    print('max seq: {}'.format(seq_max))
    print('avg seq: {}'.format(seq_avg / num_data))

    # shorten indexVector to have seq_max in sequence length
    indexVector = indexVector[:, 0:seq_max]

    # save the inputs
    matio.savemat(opt.data_path + INTERMIN_PATH + 'indexVector.mat', {'indexVector': indexVector})

    return indexVector


def get_ingre_term2word_map(ingreList, opt):
    # initialization
    wordList = []  # record the list of words in ingredients
    ingre2word_map = np.zeros((len(ingreList), 1000))
    num_words = 0  # total counts for individual words

    # create a list for ingredient words
    num_ingre = len(ingreList)
    for i in range(num_ingre):
        words = ingreList[i].split()  # get individual words in a gredient

        for word in words:
            if word in wordList:
                ingre2word_map[i, wordList.index(word)] = 1
            else:
                wordList.append(word)
                num_words += 1
                ingre2word_map[i, num_words - 1] = 1

    matio.savemat(opt.data_path + INTERMIN_PATH  + 'wordList.mat', {'wordList': wordList})

    ingre2word_map = ingre2word_map[:, 0:num_words]
    matio.savemat(opt.data_path + INTERMIN_PATH  + 'ingre2word_map.mat', {'ingre2word_map': ingre2word_map})
    return ingre2word_map, wordList


def create_glove_matrix(wordList, opt):
    # print(os.listdir(golve_root_path))
    #produce glove vectors for our ingredients
    glove_head = loadMat(opt.data_path + INTERMIN_PATH, 'glove_head.mat')
    glove_vector = loadMat( opt.data_path + INTERMIN_PATH, 'glove_vector.mat')
    num_word = len(wordList)

    p=0 #indicate the index of words in wordList
    wordVector = np.zeros((num_word,300))
    count = 0
    for word in wordList:
        print(p)
        q = 0
        for glove_word in glove_head:
            if re.match(word, glove_word):
                wordVector[p,:] = glove_vector[q,:]
                print('word {} matches glove word {}'.format(p,q))
                count+=1
                break
            q+=1
        p+=1

    print(count)
    matio.savemat(opt.data_path + INTERMIN_PATH + 'wordVector_word.mat', {'wordVector_word': wordVector})

    return wordVector

