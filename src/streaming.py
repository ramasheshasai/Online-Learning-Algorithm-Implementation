import pandas as pd

def stream_csv(path):
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        yield row
