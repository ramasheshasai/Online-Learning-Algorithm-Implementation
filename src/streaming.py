import pandas as pd
import time

def stream_csv(path, delay=0):
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        if delay > 0:
            time.sleep(delay)
        yield row
