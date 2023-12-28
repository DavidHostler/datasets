from datasets import load_dataset 
import pandas as pd
import os

imdb_dataset = load_dataset('imdb', split='train')
imdb_df = pd.DataFrame(imdb_dataset)
imdb_df.to_csv(os.getcwd() + '/imdb.csv')