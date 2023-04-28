import re
import pandas as pd
import numpy as np
def removePatterns(sentence): 
    cleaned_text  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',sentence)
    return (cleaned_text)

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def pre_process(comments_df):
    comments_df = comments_df[['cid', 'text']]
    comments_df = comments_df.dropna(axis=0, subset=['text'])
    lower = lambda x: x.lower()
    comments_df['text'] = comments_df['text'].apply(lower)

    for index, row in comments_df.iterrows():
        row['text'] = remove_emojis(row['text'])
    comments_df['text'] = comments_df['text'].replace('-', np.nan)
    comments_df['text'] = comments_df['text'].replace('  ', '', regex=True)
    comments_df['text'] = comments_df['text'].replace('"', np.nan)
    comments_df['text'] = comments_df['text'].replace('\n', '', regex=True)
    comments_df['text'] = comments_df['text'].replace(r'[\u200b\xa0]', '', regex=True)
    comments_df = comments_df.drop(index=comments_df[comments_df['text'] == ' '].index)
    comments_df = comments_df.drop(index=comments_df[comments_df['text'] == ''].index)
    pattern = lambda x: x if pd.isna(x) else re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',x)
    comments_df['text'] = comments_df['text'].apply(pattern)
    for index, row in comments_df.iterrows():
        row['text'] = re.sub(r'@\w+\s*', '', row['text'])

    cols_as_np = comments_df[comments_df.columns[1:]].to_numpy()
    sentences = []
    wh_regex = r"\b(what|where|when|why|who|whom|whose|how)\b"
    for i in cols_as_np:
        if re.search(wh_regex, i[0], flags=re.IGNORECASE):
            sentences.append(i[0])
    return sentences


