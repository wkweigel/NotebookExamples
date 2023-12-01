
#Collection of Utility Functions

#Imports
import numpy as np


#Add a column of split flags (0 or 1) to a dataframe based on a fractional split_ratio
def add_split_df_col(df, split_ratio):
    import math
    n_train = math.floor(len(df) * split_ratio)
    n_test = len(df) - n_train
    n_total = n_train + n_test
    if n_total == len(df): #if the dataframe length splits according to the split ratio evenly, proceed with out df truncation
        split_arr = np.array([1]*int(n_train) + [0]*int(n_test))
        np.random.shuffle(split_arr)
        df['split'] = split_arr
        return(df)
    else: #if the dataframe does not split evenly, proceed with df truncation
        sliced_df=df.iloc[:n_total]
        split_arr = np.array([1]*int(n_train) + [0]*int(n_test)) #Trucate the data to n=train+test
        np.random.shuffle(split_arr)
        sliced_df['split'] = split_arr
        print('The length of the df has been shortened to accomodate the split ratio')
        return(sliced_df)
    
def enumerate_binary_col(df, col_name, val1, val2):
    enumerated_vals=[]
    for i, row in df.iterrows():
        if row[col_name]==val1:
            enumerated_vals.append(-1)
        if row[col_name]==val2:
            enumerated_vals.append(1)
    return(enumerated_vals)