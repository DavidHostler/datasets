from datasets import load_dataset 
import pandas as pd
import os

# imdb_dataset = load_dataset('imdb', split='train')
# imdb_df = pd.DataFrame(imdb_dataset)
# imdb_df.to_csv(os.getcwd() + '/imdb.csv')

#Download the most relevant dataset based on the SFT task...
# imdb_dataset = load_dataset('imdb', split='train')
# imdb_df = pd.DataFrame(imdb_dataset)
# imdb_df.to_csv(os.getcwd() + '/imdb.csv')

def format_sft_df(sft_df):
  text = []
  for i in range(len(sft_df)):
    question = sft_df.iloc[i]['question']
    answer = sft_df.iloc[i]['answers']['text'][0]
    current = ' Human: ' + question + ' Assistant: ' + answer
    text.append(current)
  return pd.DataFrame({'text':text})

sft_dataset = load_dataset('lmqg/qa_harvesting_from_wikipedia', split='train')

#Convert dataset to pandas dataframe
sft_train_data = sft_dataset.to_dict()
sft_train_df = pd.DataFrame(sft_train_data)


#Format dataset for instruction tuning
sft_df_formatted = format_sft_df(sft_train_df[:50000])
print(sft_df_formatted.iloc[3]['text'])
sft_df_formatted.to_csv(os.getcwd() + '/qa_harvesting_from_wikipedia.csv')

