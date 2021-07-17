from joblib import dump

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from nepalitokenizer import NepaliTokenizer


df = pd.read_csv("../datasets/nepali/nep_news.csv")


tokenize = NepaliTokenizer()

# print(df.head)


X_train, X_test, y_train, y_test = train_test_split(df['paras'], df['label'], test_size=0.3, random_state=42)


pipe = make_pipeline(TfidfVectorizer(tokenizer=tokenize.tokenizer), LogisticRegression())

pipe.fit(X_train, y_train)


print(pipe.score(X_test, y_test))

dump(pipe, "clf.bin")

