import pandas as pd
import os


with open('data/b') as f:
    data = json.load(f)
#data = pd.read_json(os.path.join('data', 'bajs.json'))


print("finished")