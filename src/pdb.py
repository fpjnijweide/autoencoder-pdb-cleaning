import csv
import os
import re

import numpy as np
import pandas as pd
import pyAgrum as gum
import scipy
import scipy.sparse
import scipy.stats

gpu_string = ""

def natural_key(string_):
    if isinstance(string_, np.number) or isinstance(string_, bool) or isinstance(string_, int) or isinstance(string_,
                                                                                                             float):
        return string_
    else:
        """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def make_pdb(file_string, sampling_density):
    df = pd.read_csv(file_string, sep=';')
    df2 = pd.read_csv(file_string, sep=',')

    if df.columns.size < df2.columns.size:
        df = df2

    str_cols = df.select_dtypes(include=['object']).columns
    df.loc[:, str_cols] = df.loc[:, str_cols].fillna('NULL')
    df = df.where(df != 'NULL', None)

    df_cols_sorted = df.columns

    sizes_sorted = [0] * df.columns.size
    bins = [[]] * df.columns.size
    bin_widths = [None] * df.columns.size
    is_this_bin_categorical = [None] * df.columns.size

    # Determining how to bin variables
    for i in range(df.columns.size):
        coli_nonan = df.iloc[:, i][df.iloc[:, i].notnull()]
        unique_entries_in_coli = coli_nonan.unique()
        unique_entries_in_coli = np.array(sorted(unique_entries_in_coli, key=natural_key))
        # unique_entries_in_coli.sort()

        if 'categorical' in df.columns[i] or 'CATEGORICAL' in df.columns[i] or 'Categorical' in df.columns[i] or not np.issubdtype(coli_nonan.dtype, np.number):
            is_this_bin_categorical[i]=True
            sizes_sorted[i] = unique_entries_in_coli.size
            bins[i] = unique_entries_in_coli
            bin_widths[i]=None
        elif sampling_density is not None:
            # if numeric but with predefined sampling density, use that
            is_this_bin_categorical[i] = False
            bins_for_coli = np.histogram_bin_edges(coli_nonan, bins=sampling_density)
            sizes_sorted[i] = sampling_density
            bins[i] = bins_for_coli[:-1]
            bin_widths[i]=np.diff(bins_for_coli)
        else:
            # if numeric but without predefined sampling  density
            is_this_bin_categorical[i] = False
            bins_for_coli = np.histogram_bin_edges(coli_nonan, bins='auto')
            if unique_entries_in_coli.size <= bins_for_coli.size:
                # If we have mostly integers or very few numbers, just use the unique entries to determine bins
                sizes_sorted[i] = unique_entries_in_coli.size
                if len(unique_entries_in_coli)>1:
                    # calculate bin width to make PDB bins go around each number
                    bin_widths[i] = np.diff(unique_entries_in_coli)
                    bin_widths[i] = np.append(bin_widths[i],scipy.stats.mode(bin_widths[i])[0][0])
                    bins[i] = unique_entries_in_coli - 0.5 * bin_widths[i]
                else:
                    # if there is only 1 entry, don't bother calculating bin width
                    bins[i]=unique_entries_in_coli
                    bin_widths[i] = 0
            else:
                # determine bins heuristically
                sizes_sorted[i] = (bins_for_coli.size) - 1
                bin_widths[i]=np.diff(bins_for_coli)
                bins[i] = bins_for_coli[:-1]

    # TODO how about manual bin definitions?

    shape = [df.shape[0], sum(sizes_sorted)]
    sizes_sorted_with_leading_zero = [0] + sizes_sorted

    data = np.ones(df.shape[0] * df.shape[1])
    row = list(range(df.shape[0])) * df.shape[1]
    col = []

    missing_row = []
    missing_col = []
    missing_data = []
    counter = 0

    # Finding where stuff will go in the sparse matrix
    for i in range(df.values.T.shape[0]):
        for item in df.values.T[i]:
            olditem = item
            if pd.isnull(item):
                item = 0
                data[counter] = 0
                for current_bin in range(len(bins[i])):
                    prob = 1 / len(bins[i])
                    missing_col.append(current_bin + sum(sizes_sorted_with_leading_zero[0:i + 1]))
                    missing_row.append(row[counter])
                    missing_data.append(prob)
            elif not np.issubdtype(df.iloc[:, i].dtype, np.number):
                item = np.where(bins[i] == item)[0].item()
                # item = np.searchsorted(bins[i],item)
            else:
                # item = np.searchsorted(bins[i],item)
                item = np.searchsorted(bins[i], item, side='right') - 1
            newcol = item + sum(sizes_sorted_with_leading_zero[0:i + 1])
            col.append(newcol)
            counter += 1

    missing_entries_matrix = scipy.sparse.coo_matrix((missing_data, (missing_row, missing_col)),
                                                     shape=tuple(shape)).todense()
    hard_evidence_matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=tuple(shape)).todense()

    final_matrix = hard_evidence_matrix + missing_entries_matrix

    column_titles = df_cols_sorted[:]
    column_subtitles = bins

    title_subtitle_arrays = [np.repeat(column_titles, sizes_sorted),
                             [item for sublist in column_subtitles for item in sublist]]
    title_subtitle_tuples = list(zip(*title_subtitle_arrays))
    pandas_column_index = pd.MultiIndex.from_tuples(title_subtitle_tuples, names=['Variable', 'Value'])

    hard_evidence = pd.DataFrame(final_matrix, columns=pandas_column_index)

    return df, sizes_sorted, hard_evidence, bins, is_this_bin_categorical,bin_widths


def load_from_csv(input_string):
    with open(input_string, 'r') as fp:
        reader = csv.reader(fp)
        li = list(reader)
    newlist = []
    for row in li:
        newrow = []
        for entry in row[1:]:
            if entry == '':
                break
            else:
                newrow.append(float(entry))
        newlist.append(newrow)
    return newlist


def make_df(use_file, bn, mu, sigma, use_gaussian_noise, use_missing_entry, missing_entry_prob, rows, full_string,
            sampling_density, gaussian_noise_layer_sigma,output_data_string=None):
    bins = None

    if output_data_string is None:
        if not os.path.exists("./output_data/" + full_string + "/"):
            os.makedirs("./output_data/" + full_string + "/")

    if use_file is not None:
        filename_no_extension = os.path.splitext(use_file)[0]
        if sampling_density is not None:
            SD_string = str(sampling_density)
        else:
            SD_string = "None"

        try:
            # try to load the files as we might have already generated hard evidence in earlier runs (and this takes a long, long time for proper databases)
            original_database = pd.read_pickle(filename_no_extension + " SD=" + SD_string + ".df")
            hard_evidence = pd.read_pickle(filename_no_extension + " SD=" + SD_string + ".pdb")
            sizes_sorted = list(pd.read_pickle(filename_no_extension + " sizes SD=" + SD_string + ".pkl"))
            bins = list(pd.read_pickle(filename_no_extension + " bins SD=" + SD_string + ".pkl"))
            is_this_bin_categorical = list(pd.read_pickle(filename_no_extension + " categorical_bool SD=" + SD_string + ".pkl"))
            bin_widths = list(pd.read_pickle(filename_no_extension + " bin_widths SD=" + SD_string + ".pkl"))
        except:
            original_database, sizes_sorted, hard_evidence, bins, is_this_bin_categorical,bin_widths = make_pdb(use_file, sampling_density)
            if output_data_string is None:
                original_database.to_pickle(filename_no_extension + " SD=" + SD_string + ".df")
                hard_evidence.to_pickle(filename_no_extension + " SD=" + SD_string + ".pdb")
                pd.Series(sizes_sorted).to_pickle(filename_no_extension + " sizes SD=" + SD_string + ".pkl")
                pd.Series(bins).to_pickle(filename_no_extension + " bins SD=" + SD_string + ".pkl")
                pd.Series(is_this_bin_categorical).to_pickle(filename_no_extension + " categorical_bool SD=" + SD_string + ".pkl")
                pd.Series(bin_widths).to_pickle(filename_no_extension + " bin_widths SD=" + SD_string + ".pkl")

        df_cols_sorted = original_database.columns
    else:
        if output_data_string is None:
            gum.generateCSV(bn, "./output_data/" + full_string + "/" + "database_original" + gpu_string + ".csv", rows)
            original_database = pd.read_csv("./output_data/" + full_string + "/" + "database_original" + gpu_string + ".csv")
        else:
            filename_no_extension = os.path.splitext(output_data_string)[0]
            gum.generateCSV(bn, filename_no_extension + "_database_original" + gpu_string + ".csv", rows)
            original_database = pd.read_csv(filename_no_extension + "_database_original" + gpu_string + ".csv")
        original_database = original_database.reindex(sorted(original_database.columns), axis=1)


    if output_data_string is None:
        original_database.to_csv("./output_data/" + full_string + "/database_original" + gpu_string + ".csv")
    else:
        filename_no_extension = os.path.splitext(output_data_string)[0]
        original_database.to_csv(filename_no_extension + "_database_original" + gpu_string + ".csv")

    if use_file is not None:
        pass
    else:
        size_dict = {}
        for column_name in original_database.columns:
            size_dict[column_name] = bn.variable(column_name).domainSize()

        shape = [original_database.shape[0], sum(size_dict.values())]

        df_cols_sorted = sorted(list(original_database.columns))
        sizes_sorted = [size_dict[x] for x in df_cols_sorted]


        is_this_bin_categorical = list(np.array(sizes_sorted) <= 15)


        if (sampling_density <= 15):
            bin_width = 0
            bin_widths = [None for size in sizes_sorted]
        else:
            bin_width = 1
            bin_widths = [np.ones(size) for size in sizes_sorted]

        bins = [np.array(range(len(size))) - 0.5*bin_width for size in sizes_sorted]
        sizes_sorted_with_leading_zero = [0] + sizes_sorted

        data = np.ones(original_database.shape[0] * original_database.shape[1])
        row = list(range(original_database.shape[0])) * original_database.shape[1]
        col = []
        for i in range(original_database.values.T.shape[0]):
            for item in original_database.values.T[i]:
                col.append(item + sum(sizes_sorted_with_leading_zero[0:i + 1]))

        input3 = scipy.sparse.coo_matrix((data, (row, col)), shape=tuple(shape)).todense()

        first_id2 = df_cols_sorted[:]
        second_id2 = [list(range(x)) for x in sizes_sorted]

        arrays3 = [np.repeat(first_id2, sizes_sorted), [item for sublist in second_id2 for item in sublist]]
        tuples2 = list(zip(*arrays3))
        index2 = pd.MultiIndex.from_tuples(tuples2, names=['Variable', 'Value'])

        hard_evidence = pd.DataFrame(input3, columns=index2)

    if output_data_string is None:
        hard_evidence.to_csv("./output_data/" + full_string + "/ground_truth" + gpu_string + ".csv")
    else:
        filename_no_extension = os.path.splitext(output_data_string)[0]
        hard_evidence.to_csv(filename_no_extension + "_ground_truth" + gpu_string + ".csv")

    df = hard_evidence + 0

    # sigma = (sigma / sampling_density) * 100


    sigmas = []
    gaussian_noise_layer_sigmas = []
    for attribute_size in sizes_sorted:
        sigmas.append(((sigma * np.ones(attribute_size)) / attribute_size) * 100)
        gaussian_noise_layer_sigmas.append(np.ones(attribute_size) * gaussian_noise_layer_sigma(attribute_size))
        # gaussian_noise_layer_sigmas.append(gaussian_noise_layer_sigma(attribute_size))

    sigmas_per_col = np.concatenate(sigmas)
    gaussian_noise_layer_sigma_new = np.concatenate(gaussian_noise_layer_sigmas)

    noise_columns = [np.random.normal(mu, scale=s, size=(hard_evidence.shape[0])) for s in sigmas_per_col]
    noise = np.vstack(noise_columns).T

    missing_rows_clean = []
    pdb_col = 0
    for col_nr,col in enumerate(df_cols_sorted):
        missing_rows_clean_for_this_col_bool = (np.max(hard_evidence[col].values, 1) == np.min(hard_evidence[col].values, 1)) & (hard_evidence[col].shape[1] > 1)
        missing_rows_clean_for_this_col = list(missing_rows_clean_for_this_col_bool.nonzero()[0])
        missing_rows_clean += missing_rows_clean_for_this_col

        current_total_bins = sizes_sorted[col_nr]
        # pdb_col_final = pdb_col + current_total_bins - 1
        noise[missing_rows_clean_for_this_col,pdb_col:pdb_col+current_total_bins] = 0



        pdb_col += current_total_bins

    missing_rows_clean = np.unique(missing_rows_clean)

    # TODO rework how gaussian noise is added in the network, maybe need a custom layer

    if use_gaussian_noise:
        df = df + noise
        df = df.clip(lower=0, upper=1)
    if use_missing_entry:
        amount_of_variables = len(sizes_sorted)
        rows = len(df)
        total_entries = amount_of_variables * rows  # amount of probability distributions in the PDB
        missing_entry_nrs = np.random.choice(total_entries, size=round(total_entries * missing_entry_prob),
                                             replace=False)
        m = missing_entry_nrs[:]  # using an alias for shorter code
        col_index = 0
        for attribute_nr, size in enumerate(sizes_sorted):
            entries_this_col = m[(m >= rows * attribute_nr) & (m < rows * (attribute_nr + 1))]
            rows_this_col = entries_this_col - (rows * attribute_nr)
            # missing_rows += rows_this_col
            df.iloc[rows_this_col, col_index:col_index + size] = 1

            col_index += size

    missing_rows_dirty=[]

    for col in df_cols_sorted:
        df[col] = normalize_df(df[col])

        missing_rows_dirty_for_this_col_bool= (np.max(df[col].values, 1) == np.min(df[col].values, 1) ) & (df[col].shape[1] > 1)
        missing_rows_dirty_for_this_col = list(missing_rows_dirty_for_this_col_bool.nonzero()[0])
        missing_rows_dirty += missing_rows_dirty_for_this_col


    if output_data_string is None:
        df.to_csv("./output_data/" + full_string + "/noisy_data" + gpu_string + ".csv")



    missing_rows_dirty = np.unique(missing_rows_dirty)

    return df, hard_evidence, sizes_sorted, gaussian_noise_layer_sigma_new, original_database, bins, is_this_bin_categorical,bin_widths,missing_rows_dirty,missing_rows_clean


def normalize_df(df):
    newdf = df.div(df.sum(axis=1), axis=0)
    SD = len(newdf.columns)
    return newdf.fillna(1 / SD)

if __name__ == "__main__":
    # df, sizes_sorted, hard_evidence = make_pdb("surgical_case_durations.csv",None)
    df, sizes_sorted, hard_evidence, bins, is_this_bin_categorical,bin_widths = make_pdb("./input_data/Dataset - LBP RA.csv", None)
    hard_evidence.to_pickle("./input_data/surgical_case_durations.pdb")
    hard_evidence2 = pd.read_pickle("./input_data/surgical_case_durations.pdb")
    print(hard_evidence.equals(hard_evidence2))
