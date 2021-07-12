import string
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

seed = 8


contractionsDict = {"a'ight": 'alright', "ain't": 'am not', "amn't": 'am not', 'arencha': 'aren’t you', "aren't": 'are not', '‘bout': 'about', 'cannot': 'can not', "can't": 'cannot', 'cap’n': 'captain', "'cause": 'because', '’cept': 'except', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', 'dammit': 'damn it', "daren't": 'dare not', "daresn't": 'dare not', "dasn't": 'dare not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', 'dunno': "don't know", "d'ye": 'do you', "e'en": 'even', "e'er": 'ever', "'em": 'them', "everybody's": 'everybody is', "everyone's": 'everyone is', 'fo’c’sle': 'forecastle', '’gainst': 'against', "g'day": 'good day', 'gimme': 'give me', "giv'n": 'given', 'gonna': 'going to', "gon't": 'go not', 'gotta': 'got to', "hadn't": 'had not', "had've": 'had have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he had', "he'll": 'he shall', 'helluva': 'hell of a', "he's": 'he has', "here's": 'here is', "how'd": 'how did', 'howdy': 'how do you do', "how'll": 'how will', "how're": 'how are', "how's": 'how has', "I'd": 'I had', "I'd've": 'I would have', "I'll": 'I shall', "I'm": 'I am', "I'm'a": 'I am about to', "I'm'o": 'I am going to', 'innit': 'is it not', 'ion': "I don't", "I've": 'I have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it shall', "it's": 'it has', 'iunno': "I don't know", 'kinda': 'kind of', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "may've": 'may have', 'methinks': 'me thinks', "mightn't": 'might not', "might've": 'might have', "mustn't": 'must not', "mustn't've": 'must not have', "must've": 'must have', '‘neath': 'beneath', "needn't": 'need not', 'nal': 'and all', "ne'er": 'never', "o'clock": 'of the clock', "o'er": 'over', "ol'": 'old', "oughtn't": 'ought not', '‘round': 'around', "'s": 'is', "shalln't": 'shall not', "shan't": 'shall not', "she'd": 'she had', "she'll": 'she shall', "she's": 'she has', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "somebody's": 'somebody has', "someone's": 'someone has', "something's": 'something has', "so're": 'so are', 'so’s': 'so is', 'so’ve': 'so have', "that'll": 'that shall', "that're": 'that are', "that's": 'that has', "that'd": 'that would', "there'd": 'there had', "there'll": 'there shall', "there're": 'there are', "there's": 'there has', "these're": 'these are', "these've": 'these have', "they'd": 'they had', "they'll": 'they shall', "they're": 'they are', "they've": 'they have', "this's": 'this has', "those're": 'those are', "those've": 'those have', "'thout": 'without', '’til': 'until', "'tis": 'it is', "to've": 'to have', "'twas": 'it was', "'tween": 'between', "'twere": 'it were', 'wanna': 'want to', "wasn't": 'was not', "we'd": 'we had', "we'd've": 'we would have', "we'll": 'we shall', "we're": 'we are', "we've": 'we have', "weren't": 'were not', 'whatcha': 'what are you', "what'd": 'what did', "what'll": 'what shall', "what're": 'what are', "what's": 'what has', "what've": 'what have', "when's": 'when has', "where'd": 'where did', "where'll": 'where shall', "where're": 'where are', "where's": 'where has', "where've": 'where have', "which'd": 'which had', "which'll": 'which shall', "which're": 'which are', "which's": 'which has', "which've": 'which have', "who'd": 'who would', "who'd've": 'who would have', "who'll": 'who shall', "who're": 'who are', "who's": 'who has', "who've": 'who have', "why'd": 'why did', "why're": 'why are', "why's": 'why has', "willn't": 'will not', "won't": 'will not', 'wonnot': 'will not', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd've": 'you all would have', "y'all'd'n've": 'you all would not have', "y'all're": 'you all are', "y'at": 'you at', 'yes’m': 'yes ma’am', 'yessir': 'yes sir', "yesn't": 'yes not', "you'd": 'you had', "you'll": 'you shall', "you're": 'you are', "you've": 'you have', ' u ': ' you ', ' ur ': ' your ', ' n ': ' and ', ' nd ': ' and '}

custum_stopwords = {'too', 'so', 'they', 'u', 'it', 'oly', 'itself', 'yours', 'rt', 'between', 'where', 'but', 'he', 'now', 'org', 'over', 'o', 'being', 'most', 'with', 'below', 'off', 'up', 'or', 'that', 'about', "should've", 
'r', 'yourself', 'what', 'any', 'by', "you'll", 'few', "you've", 'not', 'y', 'does', 'she', "it's", 'ourselves', 'only', 'both', 'of', 'b', 'after', 'those', 'we', "that'll", 'be', 'ht', 'd', 'out', 'will', 'at', 'there', 'were',
'when', 'am', 've', 'having', 'her', 'all', 'its', 'th', 'than', 'which', 'today', "she's", 'hw', 'then', 'should', 'if', 'edu', 'and', 'the', 'as', 'n', 'through', 'them', 'also', 'because', 'further', 'against', "you're", 'how',
'who', 'was', 'some', 'own', 'before', "you'd", 'com', 'himself', 'again', 'll', 'these', 'myself', 'been', 'for', 'yourselves', 'has', 'this', 'ax', 'had', 'on', 'p', 'm', 'in', 's', 'wat', 'from', 'each', 'i', 'our', 'an',
'are', 'their', 'would', 'you', 'more', 'his', 'have', 'while', 'other', 'last', 'until', 'once', 'do', 'can', 'why', 'ours', 'just', 'like', 'doing', 'whom', 'subject', 'says', 'a', 'your', 't', 'him', 'here', 'told', 
'themselves', 'hers', 'my', 'same', 'under', 'above', 'such', 'very', 'me', 'to', 'theirs', 'during', 'is', 'ma', 'lines', 'into', 'did', 'one', 'herself', 'said', 're', 'likes'}

def expand_contractions(x):
    if type(x) is str:
        x = x.replace("\\", "")
        for key in contractionsDict:
            value = contractionsDict[key]
            x = x.replace(key, value)
        return x
    else:
        return x 

 
def preProcessing(df):
    df = df.replace(r'<ed>','', regex = True)
    df = df.replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
    df = df.replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '', regex=True)
    df = df.replace("  ", " ")
    df = df.replace(r'^(@\w+)',"", regex=True)
    df = df.replace(r'[^a-zA-Z0-9]', " ", regex=True)
    df = df.replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)
    df = df.replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)
    df = df.replace(r'^\s+|\s+$'," ", regex=True)
    df['reviews_text'] = df['reviews_text'].apply(lambda x: " ".join([token for token in x.split() if token not in custum_stopwords]))
    punctuations = string.punctuation
    df['reviews_text'] = df['reviews_text'].apply(lambda x: " ".join([token for token in x.split() if token not in punctuations]))
    df['reviews_text'] = df['reviews_text'].apply(lambda x: x.lower())
    df = df.replace({"Positive":1, "Negative":0})

    tf_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'\w{1,}',
        analyzer='word',
        ngram_range=(1,3),
        stop_words='english',
        max_features=50000,
    )
    X = df['reviews_text']
    tf_vectorizer.fit(X)
    train_word_features = tf_vectorizer.transform(X)

    model = joblib.load("xgbModel.pkl")
    preds = model.predict(train_word_features)
    df['preds'] = preds

    return df




