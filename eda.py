import pandas as pd 
import os 
import random

df = pd.read_csv(os.getcwd() + '/qa_harvesting_from_wikipedia.csv')

df = df.sample(frac=1)

print(df.head())

print(df.iloc[5]['text'])

print('SIZE: {}'.format(len(df)))