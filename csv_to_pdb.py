import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot
import scipy

import re
def natural_key(string_):
    if isinstance(string_,np.float64) or isinstance(string_,bool) or isinstance(string_,int) or isinstance(string_,float):
        return string_
    else:
        """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def make_pdb(file_string,sampling_density):
    df = pd.read_csv(file_string,sep=';')


    str_cols = df.select_dtypes(include=['object']).columns
    df.loc[:, str_cols] = df.loc[:, str_cols].fillna('NULL')
    df = df.where(df!='NULL', None) 


    df_cols_sorted = df.columns
    
    sizes_sorted = [0]*df.columns.size
    bins = [[]]*df.columns.size

    # Determining how to bin variables
    for i in range(df.columns.size):
        coli_nonan = df.iloc[:,i][df.iloc[:,i].notnull()]
        unique_entries_in_coli = coli_nonan.unique()
        unique_entries_in_coli = np.array(sorted(unique_entries_in_coli,key=natural_key))
        # unique_entries_in_coli.sort()
        if coli_nonan.dtype != np.float64:
            sizes_sorted[i] = unique_entries_in_coli.size
            bins[i] = unique_entries_in_coli
        elif sampling_density is not None:
            # use predefined sampling density
            bins_for_coli = np.histogram_bin_edges(coli_nonan,bins=sampling_density)
            sizes_sorted[i] = bins_for_coli.size-1
            bins[i] = bins_for_coli
        else:
            bins_for_coli = np.histogram_bin_edges(coli_nonan,bins='auto')
            if unique_entries_in_coli.size <= bins_for_coli.size:
                sizes_sorted[i] = unique_entries_in_coli.size
                bins[i] = unique_entries_in_coli
                bin_interval = bins[i][-1] - bins[i][-2]
                np.append(bins[i],bins[i][-1] + bin_interval)
            else:
                sizes_sorted[i] = bins_for_coli.size-1
                bins[i] = bins_for_coli

    # TODO how about manual bin definitions?

    shape = [df.shape[0], sum(sizes_sorted)]
    sizes_sorted_with_leading_zero = [0] + sizes_sorted

    data = np.ones(df.shape[0] * df.shape[1])
    row = list(range(df.shape[0])) * df.shape[1]
    col = []

    missing_row = []
    missing_col = []
    missing_data=[]
    counter = 0

    # Finding where stuff will go in the sparse matrix
    for i in range(df.values.T.shape[0]):
        for item in df.values.T[i]:
            if pd.isnull(item):
                item = 0
                data[counter] = 0
                for current_bin in range(len(bins[i])):
                    prob = 1/len(bins[i])
                    missing_col.append(current_bin + sum(sizes_sorted_with_leading_zero[0:i + 1]))
                    missing_row.append(row[counter])
                    missing_data.append(prob)
            elif df.iloc[:,i].dtype != np.float64:
                item = np.where(bins[i]==item)[0].item()
            else:
                item = np.searchsorted(bins[i],item)
            col.append(item + sum(sizes_sorted_with_leading_zero[0:i + 1]))
            counter +=1

    missing_entries_matrix = scipy.sparse.coo_matrix((missing_data, (missing_row, missing_col)), shape=tuple(shape)).todense()
    hard_evidence_matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=tuple(shape)).todense()
    
    final_matrix = hard_evidence_matrix + missing_entries_matrix

    column_titles = df_cols_sorted[:]
    column_subtitles = bins

    title_subtitle_arrays = [np.repeat(column_titles, sizes_sorted), [item for sublist in column_subtitles for item in sublist]]
    title_subtitle_tuples = list(zip(*title_subtitle_arrays))
    pandas_column_index = pd.MultiIndex.from_tuples(title_subtitle_tuples, names=['Variable', 'Value'])

    hard_evidence = pd.DataFrame(final_matrix, columns=pandas_column_index)

    return df,sizes_sorted,hard_evidence

if __name__ == "__main__":
    df, sizes_sorted, hard_evidence = make_pdb("surgical_case_durations.csv",None)
    hard_evidence.to_pickle("surgical_case_durations.pdb")
    hard_evidence2 = pd.read_pickle("surgical_case_durations.pdb")
    print(hard_evidence.equals(hard_evidence2))