from nbreversible import code

# %matplotlib inline
import pandas as pd

with code():
    df = pd.read_csv("Pokemon.csv")

with code():
    df.head()

with code():
    df.describe()
