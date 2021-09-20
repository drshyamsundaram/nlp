# NLP related resources
What is NLP ? Natural Language Processing(NLP), a field of AI, aims to understand the semantics and connotations of natural human languages. It focuses on extracting meaningful information from text and train data models based on the acquired insights. The primary NLP functions include text mining, text classification, text analysis, sentiment analysis, word sequencing, speech recognition & generation, machine translation, and dialog systems, to name a few.

The fundamental aim of NLP libraries is to simplify text preprocessing. A good NLP library should be able to correctly convert free text sentences into structured features (for example, cost per hour) that can easily be fed into ML or DL pipelines. Also, an NLP library should have a simple-to-learn API, and it must be able to implement the latest and greatest algorithms and models efficiently.

# Top NLP Libraries 
Libraries Reference: https://www.upgrad.com/blog/python-nlp-libraries-and-applications/

Natural Language Toolkit (NLTK)
Gensim
CoreNLP
spaCy
TextBlob
Pattern
PyNLPl

# 1. Natural Language Toolkit (NLTK)

NLTK is one of the leading platforms for building Python programs that can work with human language data. It presents a practical introduction to programming for language processing. NLTK comes with a host of text processing libraries for sentence detection, tokenization, lemmatization, stemming, parsing, chunking, and POS tagging.

# 2. Gensim 
Gensim is a Python library designed specifically for “topic modeling, document indexing, and similarity retrieval with large corpora.” All algorithms in Gensim are memory-independent, w.r.t., the corpus size, and hence, it can process input larger than RAM. With intuitive interfaces, Gensim allows for efficient multicore implementations of popular algorithms, including online Latent Semantic Analysis (LSA/LSI/SVD), Latent Dirichlet Allocation (LDA), Random Projections (RP), Hierarchical Dirichlet Process (HDP) or word2vec deep learning.

Gensim features extensive documentation and Jupyter Notebook tutorials. It largely depends on NumPy and SciPy for scientific computing. Thus, you must install these two Python packages before installing Gensim.

# 3. spaCy 
spaCy is an open-source NLP library in Python. It is designed explicitly for production usage – it lets you develop applications that process and understand huge volumes of text.

spaCy can preprocess text for Deep Learning. It can be be used to build natural language understanding systems or information extraction systems. spaCy is equipped with pre-trained statistical models and word vectors. It can support tokenization for over 49 languages. spaCy boasts of state-of-the-art speed, parsing, named entity recognition, convolutional neural network models for tagging, and deep learning integration.

# Why is NLP hard? 
Reference: https://www.analyticsvidhya.com/blog/2021/07/getting-started-with-natural-language-processing-using-python/

NLP is hard because natural languages evolved without a standard rule/logic. They were developed in response to the evolution of the human brain: in its ability to understand signs, voice, and memory. With NLP, we are now trying to “discover rules” for something (language) that evolved without rules.

# Understanding Textual Data 
Reference: https://www.analyticsvidhya.com/blog/2021/07/getting-started-with-natural-language-processing-using-python/

Elements of Text Let us now understand various elements of textual data and see how we can extract these using the NLTK library. Now we shall discuss the following elements of the text:

## Hierarchy of Text

Tokens
Vocabulary
Punctuation
Part of speech
Root of a word
Base of a word
Stop words

# Tokens
A meaningful unit of text is a token. Words are usually considered tokens in NLP. The process of breaking a text based on the token is called tokenization.

# Vocabulary
The vocabulary of a text is the set of all unique tokens used in it

# Punctuation
Punctuation refers to symbols used to separate sentences and their elements and to clarify meaning.

# Part of Speech
Part of speech(POS) refers to the category to which a word is assigned based on its function. You may recall that the English language has 8 parts of speech – noun, verb, adjective, adverb, pronoun, determiner, preposition, conjunction, and interjection.
Different POS taggers are available that classify words into POS. A popular one is the Penn treebank, which has the following parts of speech.
POS Image : https://editor.analyticsvidhya.com/uploads/52675tag.PNG

# Root of a word – Stemming
Stemming is a technique used to find the root form of a word. In the root form, a word is devoid of any affixes (suffixes and prefixes)

# Base of a word – Lemmatization
Lemmatization removes inflection and reduces the word to its base form

# Stop words
Stop words are typically the most commonly occurring words in text like ‘the’, ‘and’, ‘is’, etc. 
NLTK provides a pre-defined set of stopwords for English, as shown

# Frequency Distribution
Frequency distribution helps understand which words are commonly used and which are not. These can help refine stop words in a given text.

# Conditional Frequency Distribution
Conditional Frequency Distributions can help in identifying differences in the usage of words in different texts. 
For example, commonly used words in books/articles on the “romance” genre could be different from words used in books/articles of the “news” genre. 
An example with nltk library to get the conditional frequency distribution of words.
Here we use the Brown corpus. reference : https://www.nltk.org/book/ch02.html

# N-grams
N-gram is a contiguous sequence of n items from a given sample of text or speech. NLTK provides methods to extract n-grams from text

# Working with Regex Expression 
How it works A-Z Searches in the input string for characters that exist between A and Z a-z Searches in the input string for characters that exist between a and z ? Number of occurrences of the character preceding the? can be 0 or 1 . Denotes any character either alphabet or number or special characters
The number of occurrences of the character preceding the + can be at least 1 or more w Denotes a set of alphanumeric characters(both upper and lower case) and ‘_’ s Denotes a set of all space-related characters
