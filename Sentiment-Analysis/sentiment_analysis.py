'''
In this project we do sentiment analysis on Twitter data. The data consists of 5513 hand-classified
tweets and it is available at http://www.sananalytics.com/lab/twitter-sentiment/ . The objective of
the project is to classify positive and negative tweets correctly.

We first preprocess the tweets in terms of encoding, special characters, emoticons, contractions, urls etc.
We extract the features of the tweets by applying term frequency-inverse document frequency 
vectorizer. We also use a custom vectorizer which returns the average number of noun, adjectives,
verbs, adverbs, the number of exclamation marks, questions marks and the words in caps. This vectorizer
also calculates a positive and a negative score of the tweet using the scores provided at SentiWordNet
available at http://sentiwordnet.isti.cnr.it/ . After obtaining the features we feed them into a multinomial
Naive Bayes classifier. 

We do a 10-fold cross-validation on the pipeline described above. We calculate the mean accuracy and 
the F1-score on the test set. We also calculate the precision and recall values on the test
set for different threshold values as well as the area under the precision-recall curve. 

The best parameters of the Tfidf vectorizer and Multinomial Naive Bayes classifier are found by
grid search using F1-score on 10-fold cross-validation.

As a result we get the mean accuracy of the model on the test set as 0.874 and the mean
F1-score as 0.837. The mean area under the precision/recall curve is 0.90. We plot the 
the precision/recall curve for the median auc found which is 0.93.
'''

import numpy as np
import matplotlib.pyplot as plt

import csv
import HTMLParser
import re
import collections
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, precision_recall_curve, auc, f1_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator




def loadSentiWordNet(filename):
	'''
	WordNet is a large lexical database of English available at https://wordnet.princeton.edu/ .
	SentiWordNet is a lexical resource in which each WordNet synset s is associated to three numerical scores 
	Obj(s), Pos(s) and Neg(s), describing how objective, positive, and negative the terms contained 
	in the synset are. It is available at http://sentiwordnet.isti.cnr.it/
	
	Here we POS(Part of Speech) tag which corresponds to the synset type (noun, verb, adjective etc.), 
	and the synset's Positive and Negative scores. If a term occurs in multiple synsets, we take the mean of its
	positive scores and negative scores over all synsets it is found.
	'''
	
	# dictionary with key = word and value = (positive score, negative score)
	sentiment_scores = collections.defaultdict(list)
	
	infile = open(filename, 'r')
	for line in infile:	
		fields = line.split('\t')
		if fields[0].startswith('#') or len(fields) == 1:
			continue
	
		POS, ID, PosScore, NegScore, SynSetTerms, Gloss = fields
		if len(POS) == 0 or len(ID) == 0:
			continue
		
		# process each term in the synset
		for term in SynSetTerms.split(" "):
			word = term.split("#")[0]		# get the part before '#' which corresponds to the term
			word = word.replace("-", " ").replace("_", " ")
			key = "%s/%s" %(POS, word)		# write the key, e.g. 'v/understand'
			sentiment_scores[key].append((float(PosScore), float(NegScore)))	
	
	# for each term take the mean of its positive and negative scores
	for key, list_scores in sentiment_scores.items():
		sentiment_scores[key] = np.mean(list_scores, axis=0)
	
	return sentiment_scores
	

sentiment_scores = loadSentiWordNet("SentiWordNet_3_0_0_20130122.txt")



class StructuralVectorizer(BaseEstimator):
	'''
	This is a custom vectorizer which explores the structure of the text. 
	For each document it returns the average number of nouns, adjectives, 
	verbs and adverbs in the document as well as a positive score and 
	a negative score using the SentiWordNet scores. It also counts the number
	of exclamation marks, question marks and the words in all caps in the document.		 
	'''
	
	def get_feature_names(self):
		'''
		Returns the names of the features that will be calculated for each document
		after transform	
		'''
		return np.array["Positive_Score", "Negative_Score", "Avg_Nouns", "Avg_Adjectives",
						"Avg_Verbs", "Avg_Adverbs", "Num_Exclamations", "Num_Questions", "Num_Caps"]


	def fit(self, documents, y = None):
		''' 
		Fit is not applied in this setting. It just returns self.
		'''
		return self
	
	
	def _calculate_feature_values(self, document):
		'''
		Calculates the mean positive and negative score of words in the document using 
		SentiNetWord scores.
		Calculates the average number of nouns, adjectives, verbs and adverbs in the document 
		as well as the number of exclamation marks, question marks and words all in caps.  
		'''
		
		positive_scores, negative_scores = [], []
		num_nouns, num_adjectives, num_verbs, num_adverbs = 0.0, 0.0, 0.0, 0.0
		num_exclamations, num_questions, num_caps = 0, 0, 0
			
		# tokenize the document and add POS tags to each word
		words = nltk.word_tokenize(document)
		tagged_words = nltk.pos_tag(words)

			
		# count the POS types of words in the document			
		# get the positive score and the negative score for each word 
		for word, tag in tagged_words:
			positive_score, negative_score = 0.0, 0.0
			POS_type = None
				
			# check the POS type  
			if tag.startswith("NN"):
				num_nouns += 1
				POS_type = 'n'
			elif tag.startswith("JJ"):
				num_adjectives += 1	
				POS_type = 'a'					
			elif tag.startswith("VB"):
				num_verbs += 1	
				POS_type = 'v'
			elif tag.startswith("RB"):
				num_adverbs += 1	
				POS_type = 'r'
										
			# look up for the positive and negative score of the word
			# add the scores to the list of scores of the document 			
			if POS_type is not None:
				key = "%s/%s" %(POS_type, word)
				if key in sentiment_scores:
					positive_score, negative_score = sentiment_scores[key]
			
			positive_scores.append(positive_score)
			negative_scores.append(negative_score)
 			
 		# calculate the average number of nouns, adjectives, verbs and adverbs in the document
 		# calculate the mean positive and negative score of words in the document
		num_words = len(words)
		avg_nouns = num_nouns / num_words
		avg_adjectives = num_adjectives / num_words
		avg_verbs = num_verbs / num_words
		avg_adverbs = num_adverbs / num_words
		
		mean_positive_score = np.mean(positive_scores)
		mean_negative_score = np.mean(negative_scores)
 						
 				
		# count the number of exclamation marks, question marks and the words all in caps
		num_exclamations = document.count('!')
		num_questions = document.count('?')
		num_caps = np.sum([word.isupper() for word in document.split() if len(word)>2])
		
		
		# return the feature values for the document
		return np.array([mean_positive_score, mean_negative_score, avg_nouns,
						 avg_adjectives, avg_verbs, avg_adverbs, num_exclamations,
						 num_questions, num_caps])	
				
	
	def transform(self, documents, y = None):
		'''
		Returns the feature values of the documents as an array 
		'''
		return np.array([self._calculate_feature_values(document) for document in documents])			
			
			

def loadData(filename):
	'''	
	Loads the tweets and extracts positive and negative tweets
	'''
	# load the tweets and the sentiments
	infile = open(filename, "r")
	reader = csv.reader(infile)
	data = np.array([[item for item in line] for line in reader])
	X = data[1:, -1]
	y = data[1:, 1]
	
	# get the tweets with positive or negative sentiments
	# mark positive as 1 and negative as 0
	pos_neg_indices = np.logical_or(y == "positive", y == "negative")
	X = X[pos_neg_indices]
	y = y[pos_neg_indices] 
	y = (y == "positive")
	
	return X, y


	
def plot_precision_recall(precision, recall, auc_score):
	plt.fill_between(recall, precision, alpha = 0.4)
	plt.plot(recall, precision)
	plt.grid(True)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title("Precision-Recall curve with AUC = %.2f (Pos vs Neg Tweets)" %(auc_score))
	plt.show()
	


emoticons = {
	# positive emoticons
	":)" : " good ",
	":-)" : " good ",
	": )" : " good ",
	":))" : " good ",
	":-))" : " good ",
	":d" : " good ",	# lowercase representation of :D
	":-d" : " good ",
	":dd" : " good ",
	"=)" : " good ",
	";-)" : " good ",
	";)" : " good ",
	"8)" : " good ",
	"(-:" : " good ",
	"(:" : " good ",
	":>" : " good ",
	":->" : " good ",
	"x-d" : " good ",
	":'d" : " good ",
	":'-d" : " good ",
		
	# negative emoticons
	 ":(" : " bad ",
	 "=(" : " bad ",
	 ":-(" : " bad ",
	 ":/" : " bad ",
	 ":S" : " bad ",
	 ":-S" : " bad ",
	 ":'(" : " bad ",
	 ":'-(" : " bad ",
	 ":((" : " bad ",
	 ":<" : " bad ",
	 ":-<" : " bad ",
	 ":[" : " bad ",
	 ":-[" : " bad "	
}


contractions = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not"
}

html_parser = HTMLParser.HTMLParser()


def cleanText(text):

	# use utf-8
	text = text.decode("utf8")
	
	# replace html characters
	text = html_parser.unescape(text)
	
	# replace #word with word
	text = re.sub(r'#([^\s]+)', r'\1', text)
	
	# replace @user with user
	text = re.sub(r'@([^\s]+)', r'\1', text)
	
	# remove http
	text = re.sub(r"(?:\@|https?\://)\S+", "", text)
	
	# replace emoticons
	for key in emoticons.keys():
		text = text.replace(key, emoticons[key])
	
	# replace most common contractions
	for key in contractions.keys():
		text = text.replace(key, contractions[key])	
		
	return text


def preprocessText(texts):
	return np.array([cleanText(text) for text in texts])
	
	
	
def createModel(parameters = None):
	''' 
	Create a pipeline that consists of union of StructuralVectorizer 
	and TfidfVectorizer, and the Multinomial Naive Bayes.
	'''
	
	# obtain the vectorizer which combines the features from TfidfVectorizer and StructuralVectorizer 
	tfidf = TfidfVectorizer()	
	struct = StructuralVectorizer()
	union_vectorizer = FeatureUnion([('structural_features', struct), ('tfidf', tfidf)])
	
	# create the pipeline out of the union vectorizer and MultinonialNB
	clf = MultinomialNB()
	pipeline = Pipeline([("union", union_vectorizer), ("multinomialNB", clf)])
	
	# set the parameters of the pipeline components if given
	if parameters:
		pipeline.set_params(**parameters)	
	
	return pipeline		
	
	
def _gridSearch(classifier, X, y):
	'''
	Finds the best parameters for TfidfVectorizer and MultonimialNB by grid search using F1-score
	'''
	
	# set the possible values for the parameters to be examined
	parameters = dict(union__tfidf__ngram_range = [(1,1), (1,2), (1,3)],
					  union__tfidf__min_df = [1, 2],
					  union__tfidf__stop_words = ["english", None],
					  union__tfidf__smooth_idf = [True, False],
					  union__tfidf__use_idf = [True, False],
					  union__tfidf__sublinear_tf = [True, False],
					  union__tfidf__binary = [True, False],
					  multinomialNB__alpha = [0, 0.01, 0.05, 0.1, 0.5, 1])
	
	# set the K-Folds cross-validation iterator
	cv = ShuffleSplit(n =len(X), train_size = 0.8, random_state = 1)
	
	# search for the best value of C and gamma over the possible values
	grid = GridSearchCV(classifier, param_grid=parameters, cv=cv, scoring = make_scorer(f1_score), verbose = 2)
	grid.fit(X, y)
	
	return grid.best_estimator_, grid.best_params_, grid.score(X,y)*100, grid.best_score_*100



def get_best_parameters():
	''' 
	Returns the best parameters of the model found by gridSearch	
	'''
	parameters = dict(union__tfidf__ngram_range = (1,1),
					  union__tfidf__min_df = 2,
					  union__tfidf__stop_words = None,
					  union__tfidf__smooth_idf = True,
					  union__tfidf__use_idf = True,
					  union__tfidf__sublinear_tf = True,
					  union__tfidf__binary = True,
					  multinomialNB__alpha = 0.1)
	
	return parameters
	
	
	
def trainModel(classifier, X, y): 
	'''
	Does 10-fold cross-validation on the classifier. Calculates the mean accuracy and 
	the F1-score on the test set. Calculates the precision and recall values on the test
	set for different threshold values as well as the area under the precision-recall curve.  
	'''

	# obtain splitting indices of training and test sets in the ratio 80-20
	# do this split 10 times 
	cv = ShuffleSplit(n =len(X), train_size = 0.8, random_state = 7)
	
	test_accuracies = []
	precisions, recalls, f1Scores = [], [], []
	auc_precision_recall = []
	
	for train, test in cv:
		# obtain the training and test sets
		X_train, y_train = X[train], y[train]
		X_test, y_test = X[test], y[test]
		
		# fit the classifier to the training set 
		clf = classifier
		clf.fit(X_train, y_train)
		
		# get the mean accuracy of the classifier on the test set
		test_accuracy = clf.score(X_test, y_test)	
		test_accuracies.append(test_accuracy)
		
		# for each tweet in the test set get probabilities of belonging to 
		# the negative class and the positive class
		predictions = clf.predict(X_test)
		probabilities = clf.predict_proba(X_test)

		# calculate precision and recall values on the test set for different threshold values 
		proba_class_1 = probabilities[:,1]		# probability of a sample belonging to class 1
		precision, recall, _ = precision_recall_curve(y_test, proba_class_1)
		precisions.append(precision)
		recalls.append(recall)
		
		# calculate F1-score on the test set
		test_predictions = clf.predict(X_test)
		f1Score = f1_score(y_test, test_predictions)
		f1Scores.append(f1Score)
		
		# calculate the area under the precision recall curve
		auc_pr = auc(recall, precision)
		auc_precision_recall.append(auc_pr)	
	
	# report the mean accuracy on the test set
	print "The mean accuracy of the model on the test set: %.3f" %(np.mean(test_accuracies))
	print "The mean F1-score of the model on the test set: %.3f" %(np.mean(f1Scores))
	print "The mean precision/recall auc: %.3f" %(np.mean(auc_precision_recall))
	
	# plot the precision recall curve using data split that corresponds to
	# the median auc score
	median_auc_index = np.argsort(auc_precision_recall)[len(auc_precision_recall) / 2]
	plot_precision_recall(precisions[median_auc_index], recalls[median_auc_index], 
				auc_precision_recall[median_auc_index])

	
	
	
def main():
	X, y = loadData("full-corpus.csv")	
	X = preprocessText(X)
	
	model = createModel(get_best_parameters())
	trainModel(model, X, y)	

	
	
	
if __name__ == "__main__":
	main()	
