import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,ENGLISH_STOP_WORDS

dataset = pd.read_csv('English Dataset.csv')

dataset.drop('ArticleId', inplace=True, axis=1)
dataset = dataset.reset_index(drop=True)

categories = dataset['Category'].unique()

dataset_array = dataset.to_numpy()
np.random.shuffle(dataset_array)
test = dataset_array[4 * len(dataset_array) // 5:].copy()
training = dataset_array[:4 * len(dataset_array) // 5].copy()
cat_business = []
cat_tech = []
cat_politics = []
cat_sports = []
cat_entertainment = []

whole_category = [cat_business, cat_tech, cat_politics, cat_sports, cat_entertainment]

length_of_words_categorically = [0,0,0,0,0]



for i in range(len(dataset_array)):
    if dataset_array[i][1] == categories[0]:
        cat_business.append(dataset_array[i][0])
        length_of_words_categorically[0] += len(dataset_array[i][0].split())
    elif dataset_array[i][1] == categories[1]:
        cat_tech.append(dataset_array[i][0])
        length_of_words_categorically[1] += len(dataset_array[i][0].split())
    elif dataset_array[i][1] == categories[2]:
        cat_politics.append(dataset_array[i][0])
        length_of_words_categorically[2] += len(dataset_array[i][0].split())
    elif dataset_array[i][1] == categories[3]:
        cat_sports.append(dataset_array[i][0])
        length_of_words_categorically[3] += len(dataset_array[i][0].split())
    elif dataset_array[i][1] == categories[4]:
        cat_entertainment.append(dataset_array[i][0])
        length_of_words_categorically[4] += len(dataset_array[i][0].split())
print(sum(length_of_words_categorically))
row = dataset.shape[0]
# ENGLISH_STOP_WORDS = ENGLISH_STOP_WORDS.union(["said"])
from collections import defaultdict
vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
vectorizer2 = CountVectorizer(ngram_range=(2, 2),stop_words=ENGLISH_STOP_WORDS)
whole_matrix = vectorizer.fit_transform(training[:, 0]).toarray()
number_of_unique_words = whole_matrix.shape[1]
whole_matrix_of_bigrams = vectorizer2.fit_transform(training[:, 0]).toarray()
number_of_unique_bigrams = whole_matrix_of_bigrams.shape[1]
bag_of_unigrams = defaultdict()
bag_of_bigrams = defaultdict()
for c in range(len(whole_category)):
    cat_name = whole_category[c]
    X = vectorizer.fit_transform(cat_name)
    unigram_matrix = X.toarray()
    words = (vectorizer.get_feature_names_out())
    bigram_matrix = vectorizer2.fit_transform(cat_name).toarray()
    bigrams = vectorizer2.get_feature_names_out()
    vocabulary = defaultdict(int)
    vocabulary2 = defaultdict(int)
    sumOfColumns = np.sum(unigram_matrix, axis=0)
    for i in range(unigram_matrix.shape[0]):
        for j in range(unigram_matrix.shape[1]):
            if unigram_matrix[i][j] == 0:
                continue
            else:
                vocabulary[words[j]] += unigram_matrix[i][j]
    bag_of_unigrams[categories[c]] = vocabulary
    for i in range(bigram_matrix.shape[0]):
        for j in range(bigram_matrix.shape[1]):
            if bigram_matrix[i][j] == 0:
                continue
            else:
                vocabulary2[bigrams[j]] += bigram_matrix[i][j]
    bag_of_bigrams[categories[c]] = vocabulary2
    dict_word = defaultdict(int)
    print(categories[c] , '-->', end=' ')
    for i in range(50):

        max_value = np.max(sumOfColumns)

        for j in range(len(sumOfColumns)):
            if sumOfColumns[j] == max_value:
                dict_word[words[j]] = max_value

        result = np.where(sumOfColumns == max_value)[0][0]

        sumOfColumns[result] = 0
    print(*dict_word.items())
def naive_bayes(unigrams,bigrams,test):
    test = np.ndarray.tolist(test)
    import re
    global number_of_unique_words, length_of_words_categorically
    for i in range(len(test)):
        test[i] = test[i][0].lower()
        test[i] = re.sub(r'[^a-zA-Z0-9]', ' ', test[i])
        test[i] = test[i].split()
    predictions_unigram = []
    predictions_bigram = []
    for i in range(len(test)):
        scores_unigram = [0,0,0,0,0]
        scores_bigram = [0,0,0,0,0]
        for j in range(len(test[i])):
            for k in range(5):
                if test[i][j] in unigrams[categories[k]].keys():
                    scores_unigram[k] += np.log((unigrams[categories[k]][test[i][j]] +1) / (length_of_words_categorically[k] + number_of_unique_words))
                else:
                    scores_unigram[k] += np.log(1 / (length_of_words_categorically[k] + number_of_unique_words))
                if j+2 < len(test[i]):
                    bigram = test[i][j] + ' ' + test[i][j+1]
                    if bigram in bigrams[categories[k]].keys():
                        scores_bigram[k] += np.log((bigrams[categories[k]][bigram] + 1) / (length_of_words_categorically[k] + number_of_unique_words))
                    else:
                        scores_bigram[k] += np.log(1/(length_of_words_categorically[k]+number_of_unique_words))

        predictions_unigram.append(categories[scores_unigram.index(max(scores_unigram))])
        predictions_bigram.append(categories[scores_bigram.index(max(scores_bigram))])
    return predictions_unigram , predictions_bigram
def improved_naive_bayes(unigrams,bigrams,test,specific_words,absence_words):
    test = np.ndarray.tolist(test)
    import re
    global number_of_unique_words, length_of_words_categorically
    for i in range(len(test)):
        test[i] = test[i][0].lower()
        test[i] = re.sub(r'[^a-zA-Z0-9]', ' ', test[i])
        test[i] = test[i].split()
    predictions_unigram = []
    predictions_bigram = []
    for i in range(len(test)):
        scores_unigram = [0,0,0,0,0]
        scores_bigram = [0,0,0,0,0]
        for k in range(5):
            # for unique_word in set(test[i]):
              #   if unique_word in specific_words[k]:
                #     scores_unigram[k] += len(test[i]) * 0.05
                  #   scores_bigram[k] += len(test[i]) * 0.35
            for j in range(len(test[i])):
                if test[i][j] in absence_words[k]:
                    # because when we look at the final scores of each text for each category
                    # difference between min score and max score is in general about half of
                    # the length of text so if an absence word of a class is in a text
                    # we are decreasing the probability of the class but we are not doing it impossible.
                    scores_unigram[k] -= len(test[i]) * 0.5
                    scores_bigram[k] -= len(test[i]) * 3.5
                if test[i][j] in unigrams[categories[k]].keys():
                    scores_unigram[k] += np.log((unigrams[categories[k]][test[i][j]] +1) / (length_of_words_categorically[k] + number_of_unique_words))
                else:
                    scores_unigram[k] += np.log(1 / (length_of_words_categorically[k] + number_of_unique_words))
                if j+2 < len(test[i]):
                    bigram = test[i][j] + ' ' + test[i][j+1]
                    if bigram in bigrams[categories[k]].keys():
                        scores_bigram[k] += np.log((bigrams[categories[k]][bigram] + 1) / (length_of_words_categorically[k] + number_of_unique_words))
                    else:
                        scores_bigram[k] += np.log(1/(length_of_words_categorically[k]+number_of_unique_words))
        predictions_unigram.append(categories[scores_unigram.index(max(scores_unigram))])
        predictions_bigram.append(categories[scores_bigram.index(max(scores_bigram))])
    return predictions_unigram , predictions_bigram
def accuracy(outputs,predictions):
    correct_predictions = 0
    for i in range(len(outputs)):
        if outputs[i] == predictions[i]:
            correct_predictions +=1
    return 100 * correct_predictions / len(outputs)


vectorizer3 = TfidfVectorizer()
specific_word_for_each_category = []
for c in range(len(whole_category)):
    vectorizer3 = TfidfVectorizer()
    X = vectorizer3.fit_transform(whole_category[c])
    idfs = vectorizer3.idf_ - 1

    wordList = vectorizer3.get_feature_names_out()

    tfidf_matrix = X.toarray()

    specific_word = []

    sumOfCol = np.sum(X.toarray(), axis=0)

    for k in range(22):  # this range should be at least the length of the list (10)
        max_val = np.max(sumOfCol)

        for i in range(len(wordList)):

            if max_val != 0:
                if sumOfCol[i] == max_val:
                    specific_word.append(wordList[i])
                    sumOfCol[i] = 0

        if len(specific_word) == 10:
            break
    specific_word_for_each_category.append(specific_word)
    print(categories[c], ": ", specific_word, 'idf:' ,idfs)

absence_word_for_each_category = []
for c in range(len(whole_category)):
    vectorizer3 = TfidfVectorizer()
    Y = vectorizer3.fit_transform(whole_category[c])
    idfs = vectorizer3.idf_ - 1

    wordList = vectorizer3.get_feature_names_out()

    tfidf_mat = Y.toarray()

    sumOfCol = np.sum(Y.toarray(), axis=0)

    max_idf = np.max(idfs)

    min_sum = np.min(sumOfCol)

    count = 0
    absence_word = []

    for i in range(len(wordList)):
        if idfs[i] == max_idf:
            if sumOfCol[i] == min_sum:
                count += 1
                absence_word.append(wordList[i])

            if len(absence_word) == 10:
                break
    absence_word_for_each_category.append(absence_word)
    print(categories[c], ": ", absence_word, 'idf:' ,idfs)


ENGLISH_STOP_WORDS = ENGLISH_STOP_WORDS.union(
    ["said"])  # since every category has 'said', we carried 'said' to stop_word_list
non_stop_specific_for_each_category = []
for c in range(len(whole_category)):
    vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
    X = vectorizer.fit_transform(whole_category[c])
    idfs = vectorizer.idf_ - 1

    wordList = vectorizer.get_feature_names_out()

    tfidf_matrix = X.toarray()


    specific_word = []

    sumOfCol = np.sum(X.toarray(), axis=0)

    for k in range(22):  # this range should be at least the length of the list (10)
        max_val = np.max(sumOfCol)

        for i in range(len(wordList)):
            if max_val != 0:
                if sumOfCol[i] == max_val:
                    specific_word.append(wordList[i])
                    sumOfCol[i] = 0

        if len(specific_word) == 10:
            break
    non_stop_specific_for_each_category.append(specific_word)
    print(categories[c], ": ", specific_word, len(specific_word))
predictions = naive_bayes(bag_of_unigrams,bag_of_bigrams,test)
print(accuracy(test[:,1],predictions[0]))
print(accuracy(test[:,1],predictions[1]))
predictions2 = improved_naive_bayes(bag_of_unigrams,bag_of_bigrams,test,specific_word_for_each_category,absence_word_for_each_category)
print(accuracy(test[:,1],predictions2[0]))
print(accuracy(test[:,1],predictions2[1]))
predictions3 = improved_naive_bayes(bag_of_unigrams,bag_of_bigrams,test,non_stop_specific_for_each_category,absence_word_for_each_category)
print(accuracy(test[:,1],predictions3[0]))
print(accuracy(test[:,1],predictions3[1]))
for i in  range(len(predictions2)):
    if predictions2[i] == test[i][1] and predictions[i] != test[i][1]:
        print(i)









