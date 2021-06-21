import pandas as pd
import numpy as np
import scipy.sparse

def make_pdb(file_string):
    original_database = pd.read_csv(file_string,sep=';')
    # TODO what happens here? how does it handle None?
    # change nan to none, but only for strings. how about bools?
    str_cols = original_database.select_dtypes(include=['object']).columns
    original_database.loc[:, str_cols] = original_database.loc[:, str_cols].fillna('NULL')
    original_database = original_database.where(original_database!='NULL', None) 


    df_cols_sorted = list(original_database.columns)

    sizes_sorted = [pd.unique(original_database[col]).shape[0] for col in original_database.columns]

    # TODO get to hard_evidence using min/max for range of numeric, or unique strings, or bool
    # TODO make all entries numeric? 0-3 etc with titles at top
    # TODO get sizes_sorted
    # TODO what to do with None?
    sizes_sorted = None

    shape = [original_database.shape[0], sum(sizes_sorted)]
    sizes_sorted_with_leading_zero = [0] + sizes_sorted

    data = np.ones(original_database.shape[0] * original_database.shape[1])
    row = list(range(original_database.shape[0])) * original_database.shape[1]
    col = []
    for i in range(original_database.values.T.shape[0]):
        for item in original_database.values.T[i]:
            col.append(item + sum(sizes_sorted_with_leading_zero[0:i + 1]))
    # print(col[20000])

    input3 = scipy.sparse.coo_matrix((data, (row, col)), shape=tuple(shape)).todense()

    first_id2 = df_cols_sorted[:]
    second_id2 = [list(range(x)) for x in sizes_sorted]

    arrays3 = [np.repeat(first_id2, sizes_sorted), [item for sublist in second_id2 for item in sublist]]
    tuples2 = list(zip(*arrays3))
    index2 = pd.MultiIndex.from_tuples(tuples2, names=['Variable', 'Value'])

    hard_evidence = pd.DataFrame(input3, columns=index2)

    return original_database,sizes_sorted,hard_evidence

if __name__ == "__main__":
    make_pdb("surgical_case_durations.csv")