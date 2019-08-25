#Recommendation Engine
import matplotlib.pyplot as plt

import time
import numpy as np
import pandas as pd
import random
import operator
from operator import itemgetter
from statistics import mean 

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

df_ratings = pd.read_csv('df_ratings.csv' )
df_books = pd.read_csv('df_books.csv' )
df_tags = pd.read_csv('df_tags.csv' )