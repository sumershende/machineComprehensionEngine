
# coding: utf-8

# In[1]:

import pandas as pd   
import numpy as np

from bs4 import BeautifulSoup 
import re
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.porter import PorterStemmer

def clean_review(raw_review):
	example1 = BeautifulSoup(raw_review, 'html.parser').get_text()
	# print type(example1)
	# Use regular expressions to do a find-and-replace
	letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
	                      " ",                   # The pattern to replace it with
	                      example1 )  # The text to search
	# print letters_only
	lower_case = letters_only.lower()        # Convert to lower case
	words = lower_case.split()               # Split into words
	# print words
	# print stopwords.words("english")
	words = [w for w in words if not w in stopwords.words("english")]
	# print words
	from nltk.stem.porter import PorterStemmer
	porter_stemmer = PorterStemmer()
	stemmed_words = []
	for i in range(len(words)):
		stemmed_words.append(porter_stemmer.stem(words[i]))
	return( " ".join( stemmed_words ))


train = pd.read_csv("C:/Users/Ash/Documents/GitHub/nlp_bow/data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.shape)
# print train.columns.values
num_reviews = train["review"].size
clean_reviews = []
for i in range(num_reviews):
	if( (i+1)%1000 == 0 ):
		print "Review %d of %d\n" % ( i+1, num_reviews )                                                          
	clean_reviews.append(clean_review(train["review"][i]))


# In[36]:

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 1000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
print train_data_features.shape


# In[35]:

print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable

forest = forest.fit( train_data_features, train["sentiment"] )


# In[40]:

# Read the test data
test = pd.read_csv("C:/Users/Ash/Documents/GitHub/nlp_bow/data/testData.tsv", header=0, delimiter="\t",                    quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
   
    clean_test_reviews.append( clean_review( test["review"][i] ))



# In[41]:

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )






