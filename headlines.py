from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sns.set(style='darkgrid', context='talk', palette='Dark2')

reddit = praw.Reddit(client_id='<your_client_id>',client_secret='<your_client_secret>',user_agent='<your_user_name>')

headlines = set()

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)