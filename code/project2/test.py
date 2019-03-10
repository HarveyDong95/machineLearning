import pandas as pd

df = pd.DataFrame([[1,2,3],[4,5,6]])

list1 = [7,8,9]

df.loc[2] = list1
