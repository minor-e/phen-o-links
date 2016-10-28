#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written and inspired by Esteban Fernandez Parada
# Date for project started:12 June 2014
# This is an alpha mode version of script phen_o_links and its main goals.


import pandas as pd
import numpy as np
import re
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as distance
from collections import defaultdict
from pprint import pprint
from scipy.stats import ttest_ind
from scipy.stats import linregress
from scipy.stats import ttest_ind_from_stats
from os import listdir
from os.path import abspath
from os.path import exists
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

# Helper functions


def dataset_sign_checker(df, df2, columns=[], columns2=[]):
    """The function compares value sign for a given column label.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called "df" is the main data set that compare to df2.

    df2 : pandas.core.frame.DataFrame(object)
        The parameter called "df2" is the data set that is used to compare main
        data set called "df".

    columns, columns2 : list(object)
        The parameter called "columns" and "columns2" is a list object that
        accepts string entries. If left empty interactive mode is triggered.

    Returns
    --------
    df_main : pandas.core.frame.DataFrame(object)
        The "df" with three new columns. The sign from the 'df' values is
        labelled as "Sign_df", and the sign values for 'df2' is called
        'Sign_df2' and lastly the third column is called "Boolean_Compare". The
        "Boolean_Compare" is a truth table between 'Sign_df' and 'Sign_df2'.
    """
    # Copying frame
    df_work1 = dataset_copy_frame(df)
    df_work2 = dataset_copy_frame(df2)

    # Check columns & columns2 is empty

    try:
        if len(columns) + len(columns2) == 0:
            text = ("columns and columns2 need to be have value")
            raise Warning(text)
    except(Warning):
        text2 = ("Calling function to pick values for columns\n")
        print text2
        columns, index = dataset_pick_columns(df_work1, split="groupby")
        columns2, index2 = dataset_pick_columns(df_work2, split="groupby")

    # Creating values signs
    df_work1["df_Sign"] = np.sign(df_work1[columns[0]].values)
    idx = np.sign(df_work2[columns2[0]].values)
    df_work1["df2_Sign"] = idx

    # Checking if signs values are identical
    df_work1["Boolean_Compare"] = df_work1.df_Sign.values == idx

    # Returning modified table
    return df_work1


def dataset_create_table(data=[], columns=[], index=[]):
    """ Creates a simple table from pandas data frame core objects e.g.
    after using filtering function look at lost.
    Remember that all column names must have a corresponding value.

    Parameters
    ----------
    data : list(optional)
        The 'data' parameter accepts pandas.core.frame.DataFrame(object)
        or other types of data types.

    columns, index : list(optional)
        The accepts string or integers types items.

    Returns
    -------
    table : pandas.core.frame.DataFrame(object)
        The 'table' is pandas data frame object.

    Raises
    ------
    ValueError
        If 'data' and 'columns' differ in length.
    """

    # Check if columns and data has the same length
    if not(len(data) == len(columns)):
        prompt = ("\n Parameters 'data' and 'columns'"
                  " must be the same length."
                  " Given length for:\n'data' = {0}"
                  " \n 'columns' = {1}").format(len(data), len(columns))
        raise ValueError(prompt)
        return

    # Creating dictionary for values.
    data_dict = dict(zip(columns, data))

    # Creating table
    table = pd.DataFrame(data=data_dict, columns=columns, index=index)

    return table


def dataset_getting_control_positions(colony_format, position=[0,0,0,0]):
    """The function returns the nth- position of every colony for a given
    format.

    Parameters
    ----------
    colony_format : int(object)
        The parameter called 'format' takes any array shape (12,8)
        and creates a numpy array.

    position : list(object)
        The parameter called 'position' is list object that represent a 2 by 2
        quadrant. Accepted entries in 'position' parameter is 1 typed as int.
        The 'position' has total length of 4, where each indices is location
        starting 1th = top left corner of quadrant and 4th = at bottom right
        corner of quadrant.

    Returns
    -------
        df_ctrl : pandas.core.frame.DataFrame(object)
        The return value called 'df_ctrl' is a pandas object with
        two columns called 'Rows' and 'Columns', which are the plate
        coordinates for the control colonies.

    Raises
    ------
    ValueError
        If 'colony_format' not divisible by 96 and or shape of array (12,8)
        dimension not followed!

        If 'position' contains item value that is either not typed as int
        or if value entered is greater than 1.
    """

    # Local Global
    coordinates = []
    control_position = []

    # Creating plate format
    plate = dataset_create_coordinate_system(colony_format)

    # Check entries

    try:
        assert(len([isinstance(i, int) for i in position if i <= 1]) == 4)

    except AssertionError:
        text = ("The parameter called 'position' has"
                " an invalid entry please enter either"
                " 0 or 1 integers in 'position' parameter!")
        raise ValueError(text)

    # Creating control array
    # Check for index numbers and slice accordingly:
    # x 0 or 0 x
    # 0 0    0 0
    if position.index(True) <= 1:
        slice_mode = position.index(True) % 2
        control_position = plate[0::2, int(slice_mode)::2]
    # Check for index numbers and slice accordingly
    # 0 0 or 0 0
    # x 0    0 x
    if position.index(True) > 1:
        slice_mode = position.index(True) % 2
        control_position = plate[1::2, int(slice_mode)::2]

    # Getting coordinates
    for i in control_position.ravel():
        coordinates.append(np.where(plate == i))

    # Creating pandas object
    df_ctrl = pd.DataFrame(coordinates, columns=('Rows', 'Columns'))
    df_ctrl.Rows = df_ctrl.Rows.astype(int)
    df_ctrl.Columns = df_ctrl.Columns.astype(int)

    return df_ctrl


def dataset_subsample_control_colonies(df, df_ctrl):
    """The function subsamples control colonies from main dataset
    minimum requirements for function to run are that both data
    frame inputs have a column category called 'Columns' and 'Rows'.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' is the main
        dataset from which subsample(s) are drawn.

    df_ctrl : pandas.core.frame.DataFrame(object)
        The parameter called 'df_ctrl' is the return value
        from data_set.dataset_getting_control_positions function.

    Returns
    -------
    df_control : pandas.core.frame.DataFrame(object)
        The return value called 'df_control' contains
        all control positions and values.

    df_main : pandas.core.frame.DataFrame(object)
        The return value called 'df_main' is the main
        dataset 'df' without control position.

    See Also
    --------
        data_set.dataset_getting_control_positions : for more information about input
                                                     'df_ctrl'.

    """

    # Copying frames

    df_work1 = dataset_copy_frame(df)
    df_ctrl1 = dataset_copy_frame(df_ctrl)

    # Instructing user!
    text = ("\n\nPick column category that "
            "contains 'Rows' and 'Columns'\n\n")
    print text

    print "\nFor main dataset first 'df'!\n "
    main_column, index = dataset_pick_columns(df_work1, split='groupby')
    print "\nFor dataset called 'df_ctrl'!\n"
    ctrl_column, index = dataset_pick_columns(df_ctrl1, split='groupby')

    del index

    try:
        assert(len(main_column) == 2 and len(main_column) == 2)

    except AssertionError:
        text = ("'Columns' picked by"
                " user is either <2 or >2 for 'df' or 'df_ctrl'!")
        raise ValueError(text)

    # Creating coordinates series

    main_coords = pd.Series(
        zip(df_work1[main_column[0]].values, df_work1[main_column[1]]))
    ctrl_coords = pd.Series(
        zip(df_ctrl1[ctrl_column[0]].values, df_ctrl1[ctrl_column[1]]))

    # Using series to index main frame by boolean value
    df_work1['Control_Pos'] = main_coords.isin(ctrl_coords)

    # Slicing main frame
    df_main = df_work1[df_work1['Control_Pos'] == False]
    df_control = df_work1[df_work1['Control_Pos'] == True]

    # Returning values!
    return df_control, df_main


def dataset_check_by_Global_Std(df, columns=[], indexer=[], aleph=0.05):
    """Function assume that data is normal distributed, that standard deviation
    is known and that samples are drawn at random independent from each other.
    Requires columns names to end with 'var' and contain the mean variance
    for each labels, which is used for mean for standard
    deviation calculations.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        With new boolean column for values that passes upper or lower limit
        for given certain confidence interval.

    columns : list()
        The 'columns' parameter takes string inputs, which are column names.
        The 'columns' should be order in the following matter 'Experimental
        sample mean values', 'Control sample mean values', 'Experimental sample
        mean variance' and 'Control sample mean variance'.

    indexer : list()
        The 'indexer' is obsolete and not used for any thing. But still need
        a value due to function call

    aleph : float
        The 'aleph' parameter is almost the alpha value or the type I error
        of given a normal distribution applying the sigma rule of 3
        , where sigma is the global standard deviation calculated from
        mean variances of sample given in 'columns'.
        The 'aleph' can either be set to 0.05 or 0.01 default value
        is 0.05.

    Returns
    -------
    df_work1 : pandas.core.frame.DataFrame(object)
        The 'df_work1' is the given data frame given called 'df'
        with 2 or more additional columns. One is field with boolean
        values and answer if the difference between mean samples >
        number of global standard deviation. The 2nd type is a column
        title Ratio_xxx, which the difference between mean sample values
        divided by number of global standard deviations.

    Raises
    ------
    ValueError
        If column names given are not present or if columns with mean sample
        variance are not flagged with 'var' at the end.
        If aleph is the 0.05 or 0.01.

    See Also
    --------
    data_set.dataset_pick_columns : For more information about 'indexer' and
                                    'split' parameters.

    References
    ----------
    .. [1] Wikipedia contributors, '68–95–99.7 rule', Wikipedia, The Free
           Encyclopedia, 26 June 2016, 21:11 UTC
    """
    # Local Global
    nr_std = 2

    # Coping input frame.
    df_work1 = dataset_copy_frame(df)

    # Making sure that values for calculations are given.
    if not (columns and indexer):
        columns, indexer = dataset_pick_columns(df_work1, split='groupby')

    # Check if columns and indexer item(s) present in data frame.
    try:
        [df_work1.columns.tolist().index(i) for i in columns]
        [df_work1.columns.tolist().index(y) for y in indexer]

    except ValueError:
        print ("Input values was \n{0} \n\n{1} and"
               " values present in frame"
               " is{2}").format(i, y, df_work1.columns.tolist())
        return

    # Returns string with var at end for mean variance per label
    variance_columns = [i for i in columns if i.endswith('var')]

    # Check that input frame contains at var
    try:
        if not variance_columns:
            raise ValueError

    except ValueError:
        print ("Present column names lack 'var' flag at the"
               " which is used for standard deviation calculation."
               " Columns found with var flag was zero!"
               "Please flag sample mean variances with 'var' at"
               " the end of column name(s)")
        return

    # Check that aleph is either 0.05 or 0.01
    try:
        if not (aleph == 0.05 or aleph == 0.01):
            raise ValueError

    except ValueError:
        print ("Parameter called aleph must be either 0.05 or 0.01."
               " \nUser input was:{0} ").format(aleph)
        return

    # Conditional that sets the number of stds to multiply with.
    if aleph == 0.01:
        nr_std = 3

    # New column names
    new_names = ['Global_' + i.replace('_var', ('_std_X_' + str(nr_std)))
                 for i in variance_columns]
    new_names2 = [i.replace('Global_', 'Ratio_')
                  for i in new_names]
    # sqrt((abs(Experiment) - abs(Control))**2) > nr_std * standard deviation
    for i in range(len(variance_columns)):
        df_work1[new_names[i]] = np.sqrt((
            np.abs(df_work1[columns[0]].values) -
            np.abs(df_work1[columns[1]].values)) ** 2) > np.sqrt(df_work1[
                variance_columns[i]].mean()) * nr_std
            # Adding effect size ratio column(s).
        df_work1[new_names2[i]] = (np.sqrt((
            np.abs(df_work1[columns[0]].values) -
            np.abs(df_work1[columns[1]].values)) ** 2)) / (np.sqrt(df_work1[
                variance_columns[i]].mean()) * nr_std)
    return df_work1


def dataset_filter_by_pvalues_and_Global_std(
        df, columns=[], indexer=[], alpha=0.05):
    """ The function filters a data frame with 2 parameters, a boolean columns
    for global standard deviation and raw p-values column that does not
    exceed alpha value.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The 'df' must be the return from dataset_check_by_Global_Std function.

    columns : list(object)
        The 'columns' is a list object containing the string names for
        the boolean returns for difference between experimental
        and control values > Global standard deviation.
        If parameter left empty, it trigger the dataset_pick_columns
        function call.

    indexer : list(object)
        The 'indexer' is a list object where indices are
        column names for **raw** p-values.
        If parameter left empty, it trigger the dataset_pick_columns
        function call.

    alpha : float(object)
        The 'alpha' is the set p-value to filter data set 'df' with.
        The parameter 'alpha' is set to '0.05' as default value.

    Returns
    -------
    subs : list(object)
        The 'subs' is list object with sub data frame(s) from
        the given 'df'.

    See Also
    --------
    dataset_pick_columns : For more information about 'columns' & 'indexer'.
    dataset_check_by_Global_Std : For more information about boolean columns.
    dataset_pvalue : For more information about raw p-values.

    Notes
    -----
    If indices for 'columns' and 'indexer' differ function assumes that the
    values present in 'indexer' should be added multiple times until difference
    in indices disappear between 'indexer' and 'columns'.

    """
    # Local Global
    subs = []
    indexer2 = []

    # Copy data frame
    df_work1 = dataset_copy_frame(df)

    # Making sure that values for calculations are given.
    if not (columns and indexer):
        print ("\nThe 1st round of names picking is the boolean columns for"
               " global standard deviation.\n"
               "\nThe 2nd round which raw p values to filter from called"
               " indexer here.")
        columns, indexer = dataset_pick_columns(df_work1, split='groupby')

    if not (len(columns) == len(indexer)):
        print "*" * 100
        print ("\nWarning: Assuming that the same P-values column"
               " is used for filtering boolean in 'columns'!\n")
        # Appending values to indexer2.
        while len(indexer2) < len(columns):
            indexer2.append(indexer[0])
        indexer = indexer2

    # Loop that appends subs set from input frame.
    for i in range(len(columns)):
        tmp = df_work1[
            (df_work1[columns[i]].values == True) & (df_work1[
                indexer[i]].values < alpha)]
        subs.append(tmp)

    return subs


def dataset_fill_NAN_with_any_value(df, mask='', columns=[]):
    """ Function fills NaNs in data frame with a given value.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input pandas data frame object, with NaN values.

    mask : str, float, int
        Masked value to replace NaNs in data frame df.

    columns : list(object)
        The parameter called 'columns' accepts string elements
        as input and input should be present in 'df' as column
        name.

    Returns
    -------
    df_work1 : pandas.core.frame.DataFrame(object)

        The new copy of 'df' where NaNs have been masked by 'mask'
        value.

    Raises
    ------
    ValueError
        If 'mask' parameter is wrong typed or left empty.
        If string in put in columns not present in 'df'
        column names.
    """
    # Copy frame
    df_work1 = dataset_copy_frame(df)

    # Local global
    columns_in_df = df_work1.columns.tolist()

    # Check that mask is not empty or list(object)

    if not mask or type(mask) == list:
        print (" \n\nParameter empty or type"
               " in put in mask is not valid"
               " value entered for 'mask':\n"
               " \n\t {0}"
               " and the type for 'mask' was \n"
               " \n\t {1} . ").format(mask, type(mask))
        raise ValueError

    # Automatic search not specified by column str mask value

    if type(mask) == str and not columns:
        df_work1 = df_work1.replace(np.nan, mask, regex=True)
        return df_work1

    # No columns specified search take all present column names.
    if not columns:
        columns = df_work1.columns.tolist()

    # Check for that 'columns' items are present in frame
    try:
        for i in columns:
            columns_in_df.index(i)
    except ValueError:
        print (" Items given in columns was: \n\t{1}"
               " and columns present in 'df' is:"
               "\n\t {0}").format(columns_in_df, columns)
        return

    # Iterate through items in columns parameter.
    for i in columns:
        # Check if columns contains NaNs.
        if df_work1[i].isnull().sum():
            if type(mask) != str:
                # NaNs replaced with mask value
                df_work1[i] = df_work1[i].fillna(value=mask)
            else:
                # NaNs replaced with mask value
                df_work1 = df_work1.replace({i: np.nan}, {i: mask})

    # Returns new frame with nan-filled
    return df_work1


def dataset_filter_by_p_values_and_fold_change(
        df, alpha=0.05, fold_change=2.0, work_columns=[]):
    """ The function returns a subset from input data frame
    based on the criteria given by user.
    All values given to 'fold_change' must be unlogged and log2 transformable.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input frame with adjusted p-values, frame
        returned from data_set.dataset_p_values_adjustmentdata_set.

    alpha : float(optional)
        The 'alpha' parameter determines the type I error it's set
        to '0.05'.

    fold_change : float(optional)
        The 'fold_change' parameter is used for filtering
        by ratios between reference set and experimental set.
        The value given to 'fold_change' is log2() in function.
        The 'fold_change' is set to '2', which filters by ratio value
        ratio>=2 or or ratio <= 0.5 in none log-scale.

    work_columns : list(object)
        The 'work_columns' parameter has a total of 3 string items the order
        ith-item is important.First item is the string name for adjusted
        p-values, the second item is the reference set and third item is
        the experimental set. The order for 'work_columns' is P-values adjusted,
        Ref-set and Experimental-set.

    Returns
    -------
    df_subset : pandas.core.frame.DataFrame(object)
        The return frame called 'df_subset' is a subset from enter
        'df', where values have been filtered by p-adjustments and
        fold_change.

    Raises
    ------
    IndexError
        If 'work_columns' contains more or less then 3 items.

    ValueError
        If item in 'work_columns' is not found in 'df' frame.
        If 'df_subset' returns an empty frame.

    """
    # Global local
    log_fold_change = np.log2(float(fold_change))
    column_names = df.columns.tolist()

    # Copying input frame
    df_work1 = dataset_copy_frame(df)

    # Check for column names
    try:
        work_columns[2]

    except IndexError:
        print ("\n Parameter called 'work_columns' lacks or has more"
               " than 3 items in list. "
               "\nUser entered: \n\t {0}.").format(work_columns)
        return
    try:
        [column_names.index(i) for i in work_columns]

    except ValueError:
        print ("\n Please enter one or more valid option 'work_columns': "
               "\n\t {0} and "
               "\n\n User entered: \n\t {1}.").format(column_names, i)
        return

    # Filtering by p-values
    df_alpha = df_work1[df_work1[work_columns[0]] <= alpha]

    # Calculation for 'Log_fold_change'.
    df_alpha['Log_Fold_Change'] = (df_alpha[work_columns[2]].values -
                                   df_alpha[work_columns[1]].values)
    # Filtering by Log_fold_change
    df_subset = df_alpha[(
        df_alpha['Log_Fold_Change'] <= np.negative(log_fold_change))
        | (df_alpha['Log_Fold_Change'] >= log_fold_change)]

    # Check data frame is not empty
    if not df_subset.shape[0]:
        print ("\n\n Frame empty please try"
               " another p-values adjustment or data frame")
        raise ValueError
        return
    # Returns subsets
    return df_subset


def dataset_indices_maker_for_slicing_dataframes(df):
    """ Function returns a 2 column by ith-rows array. The array
    is usable with pandas.DataFrame.loc or other slicing function
    and it's an integer based slicing.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The input data frame wanted to be sliced.

    Returns
    -------
    array_shaped : np.array
        The 'array_shaped' has the indices wanted for splitting data.
    """
    # Copy frame.
    df_work1 = dataset_copy_frame(df)

    # Counting number of items in frame.
    indicies_nr = np.arange(len(df_work1))

    # Creating slicing indices in array.
    array_sliced = indicies_nr[:len(indicies_nr) - (len(indicies_nr) % 2)]

    # Creating array rows.
    row_dimension = (len(indicies_nr) - (len(indicies_nr) % 2)) / 2

    # Creating a 2X2 array.
    array_reshaped = array_sliced.reshape(row_dimension, 2)

    # Check if extra row is needed.

    if len(indicies_nr) % 2:
        # Calculating indices for new_row.
        new_row = [
            array_reshaped[len(array_reshaped) - (len(indicies_nr) % 2)][1],
            array_reshaped[len(array_reshaped) - 1][1] + (len(
                indicies_nr) % 2)]
        # Adding new row to array.
        array_reshaped = np.vstack([array_reshaped, new_row])

        # Returning array 2Xy array.
        return array_reshaped

    # Returning array 2Xy array.
    return array_reshaped


def dataset_p_values_adjustment(df, work_column=[], p_method='BY'):
    """ The function returns a data frame with where p-values been adjusted
    according to p-value adjustment method. The function works on 1x1 biases,
    1 column to work with and 1 new column as output.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' must contain a column containing p-values.

    work_column : list(optional)
                The parameter called 'work_column' is list type and accepts
                only a string entries. The 'work_column' is the name for column
                containing p-values.

    p_method : str(optional)
            The parameter called 'p_method' is the p-value adjustment algorithm
            used for the given p-values in work_column. Valid p-value
            correction entries are methods 'holm', 'hochberg', 'hommel',
            'bonferroni', 'BH','BY', 'fdr' and 'none'. The value for 'p_method'
            is set to 'BY', which is 'Benjamini–Hochberg-Yekutieli' procedure
            of correcting p-values.

    Returns
    -------
    df_work1 : pandas.core.frame.DataFrame(object)
              The same entered 'df input with the addition of a new column with
              p-values corrected for specified method.

    Raises
    ------
    ValueError
        If string entered in 'p_method' and 'work_column' is not found and
        if 'work_column' entries is greater than 1 and is of wrong type. Also if
        data frame return is empty!
    """

    # List of valid methods
    valid_methods = ['holm', 'hochberg', 'hommel', 'bonferroni',
                     'BH', 'BY', 'fdr', 'none']
    try:
        valid_methods.index(p_method)
    except ValueError:
        print ("\nPlease enter a valid option for p_method: "
               "\n\t {0} and "
               "user entered: \n\t {1}.").format(valid_methods, p_method)
        return
    # Copy frame
    df_work1 = dataset_copy_frame(df)
    column_names = df_work1.columns.tolist()

    # Check for work_column in df
    if not work_column[0] in column_names:
        print ("\n Column named \n\t {0} \n not found."
               "\n\n Please pick a column name present "
               "in \n {1}.\n").format(work_column[0], column_names)
        raise ValueError
        return

    if len(work_column) > 1 or type(work_column[0]) is not str:
        print ("\n Column named {0} has more than 1 entry {1} "
               "or type entered in work_column is not a"
               "string {2}").format(
                   work_column[0],
                   len(work_column),
                   type(work_column[0]))
        raise ValueError
        return

    # Sort in ascending order for work_column
    df_work1 = df_work1.sort(columns=[work_column[0]])

    # New column name created
    new_column = str(work_column[0]) + '_method_' + str(p_method)

    # Using rpy2 module
    stats_tool = importr('stats')

    # Adding new column to frame with adjusted p-values!

    df_work1[new_column] = stats_tool.p_adjust(
        FloatVector(df_work1[work_column[0]].values), method=p_method)

    # Check data frame is not empty
    if not df_work1.shape[0]:
        print ("\n\n Frame empty please try"
               " another p-values adjustment or data frame")
        raise ValueError
        return

    # Returning frame with adjusted p-values
    return df_work1


def dataset_normalise_values_heatmap(df):
    """Function returns the max value and its negative correspondent by
    multiplication with -1 in a pandas data frame object.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter 'df' is a pandas data frame object.

    Returns
    -------
    vmin, vmax : int
        The return value is the smallest and the
        greatest integer found in 'df' parameter.

    Raises
    ------
    TypeError
        If 'df' parameter is not pandas data frame object!
    """
    df_work1 = dataset_copy_frame(df)
    try:
        if type(df_work1) is not type(pd.DataFrame()):
            raise TypeError
    except TypeError:
        print ("The 'df' input was not a pandas data frame object. "
               "User gave following:\n {0}").format(type(df_work1))
        return
    vmin = np.floor(df_work1.min().min())
    vmax = np.ceil(df_work1.max().max())
    vmax = np.max([vmax, np.abs(vmin)])
    vmin = vmax * -1
    return vmin, vmax


def dataset_pairwise_square_distance_heatmap(df):
    """Takes a data frame object and returns the square pairwise distance
    for rows and columns.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The 'df' is a given pandas data frame object.

    Returns
    -------
    row_distance, column_distance : numpy.ndarray
    The return values 'row_distance' and 'column_distance'
    are numpy array with square pairwise distance of df data.

    Raises
    ------
    AttributeError
        If 'df' parameter is not a pandas data frame object.
    """
    # Copying frame
    df_work1 = dataset_copy_frame(df)

    # Check that inputs is correct
    try:
        if not isinstance(df_work1, pd.DataFrame):
            raise AttributeError
    except AttributeError:
        print ("Input given to function was not "
               "a pandas.core.frame.DataFrame object. User "
               "input following type:\n {0} ").format(type(df_work1))
        return

    # Calculation of of square distance
    row_distance = distance.squareform(distance.pdist(df_work1))
    column_distance = distance.squareform(distance.pdist(df_work1.T))

    return row_distance, column_distance


def dataset_cluster_data_heatmap(data=[], calc_method='complete'):
    """The function returns a cluster for given set of data.

    Parameters
    ----------
    data : list
        The parameter called 'data' is an empty list object.

    calc_method : str(optional)
        The 'calc_method' is the algorithm of choice to produce
        cluster(s).

    Returns
    -------
    cluster : list
        The return value 'cluster' is list object where items order
        follows the input data's order.

    Raises
    ------
    IndexError
        If 'data' parameter has is left empty.
    TypeError
        If 'data' items are not numpy.ndarray objects.

    See Also
    --------
        scipy.cluster.hierarchy : For more information about valid
        string entries for parameter called 'calc_method'
        argument called 'method' in scipy.cluster.hierarchy.
    """
    # Global local variable
    cluster = []

    # Checking that data is not empty
    try:
        if not data:
            raise IndexError
    except IndexError:
        print "The parameter 'data' is empty please insert data!"
        return

    # Adding to cluster variable
    for i in range(len(data)):
        # Checking that type is correct
        try:
            if type(data[i]) is not type(np.ndarray(0)):
                raise TypeError
        except TypeError:
            #index = [type(data[i]) != type(np.ndarray())]
            print ("Input parameter 'data' "
                   "has the wrong type. User "
                   "input was:\n {0} "
                   "at list index {1}").format(type(data[i]), i)
            return
        temp_storer = sch.linkage(data[i], method=calc_method)
        cluster.append(temp_storer)
    return cluster


def dataset_retreving_coordinates_to_heatmap(df):
    """The function returns coordinates for heat map plotting and
    scoring for position.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The parameter called df is pandas data frame, where column
        labels 'Rows' and 'Columns' exist.

    Returns
    -------
    heatmap_coordinates : list(tuple pairs)
        The return value for function is a list with tuple pairs for
        items.

    Raises
    ------
    AssertionError
        If column labels 'Rows' and 'Columns' not found in data frame.

    """
    # Copying frame
    df_work1 = dataset_copy_frame(df)

    # Getting columns in data frame
    columns_list = df_work1.columns.tolist()

    # Test that columns Rows & Columns are present
    try:
        assert 'Rows' and 'Columns' in columns_list

    except AssertionError:
        print ('\nData frame has no column label called '
               'Rows or Columns ')
        return

    heatmap_coordinates = [(
        df_work1.Rows.values[i],
        df_work1.Columns.values[i]) for i in range(
            len(df_work1.Rows.values))]
    # Returns list with ready to go coordinates.
    return heatmap_coordinates


def dataset_joins_two_dataframes(
        df_1, df_2, combine_as='left',
        assemble_method='drop'):
    """The function joins two data frame object if
    a common index value is found for both frames.

    Parameters
    ----------
    df_1, df_2 : pandas.core.frame.DataFrame(object)
        The parameters called 'df_1' and 'df_2' are
        pandas data frame objects. The parameters must be
        entered in the following matter df_1 >= df_2.

    combine_as : str(optional)
        The parameter called 'combine_as' is string and accepts the
        same valid entries as the join method in pandas.
        Parameter is set to 'left' which is equivalent to
        left join in SQL.

    assemble_method : str(optional)
        The parameter called 'assemble_method' is string entry
        and has 2 valid entries 'drop' or 'mean'. The parameters
        either drops duplicated entries or means duplicated entries
        found in data frames. Parameter default set to 'drop'.

    Returns
    -------
    df_combine : pandas.core.frame.DataFrame(object)
        The new formed data frame created from 'df_1' and 'df_2' data.

    Raises
    ------
    IndexError
        if df_1 < than df_2 entry in indices.
    NameError
        If indexes values given for 'df_1' and 'df_2' do not match.
    ValueError
        if 'assemble_method' parameter has other values then
        the valid ones 'drop' or 'mean'.
    """
    # Local Global
    split = 'groupby'

    # Testing df_1 bigger or equal to df_2
    try:
        if not len(df_1) >= len(df_2):
            raise IndexError

    except IndexError:
        print ("The amount of entries "
               "for 'df_1' < 'df_2' please "
               "enter data frames in following "
               "matter:\n 'df_1' >= 'df_2'")
        return

    # Fixing frames to be combined and layout.
    df_1 = dataset_clear_columns(df_1)
    df_2 = dataset_clear_columns(df_2)

    df_work1 = dataset_copy_frame(df_1)
    df_work2 = dataset_copy_frame(df_2)

    # Pick column label to assemble
    print ("\n Pick 1 column from df_1 "
           ", that function as index to "
           "add data to df_1 !\n")
    cols_1, index_1 = dataset_pick_columns(df_work1, split=split)

    print ("\n Pick 1 column from df_2 "
           ", that has similar index values as the "
           "one picked for df_1 expansion!\n")
    cols_2, index_2 = dataset_pick_columns(df_work2, split=split)

    # Check that user pick 1 column label per section
    if not (len(cols_1) + len(cols_2)) == 2:
        print ("User should only pick 1 column per frame "
               "'df_1' and 'df_2' user entered:\n "
               "'df_1' :\t {0} , 'df_2' :\t {1} ").format(cols_1, cols_2)
        return

    # Check that indexes picked are valid
    count = sum([
        df_work2[[cols_2][0]].values[i] in df_work1[[cols_1][0]].values
        for i in range(len(df_work2[cols_2[0]].values))])
    try:
        if count <= 0:
            raise NameError

    except NameError:
        print ("Columns picked do not contain similar values "
               "and therefor data frames cannot be combine! "
               "Please pick column labels with similar values.")
        return

    # Logic circuit on how to deal with duplicated instances
    assemble_method = assemble_method.lower()

    # Test that assemble_method is correct
    try:
        if not (assemble_method == 'drop' or assemble_method == 'mean'):
            raise ValueError

    except ValueError:
        print ("The parameter called 'assemble_method' "
               "has only 2 valid entries 'drop' or 'mean'. "
               "User entered: \n {0}").format(assemble_method)
        return

    if assemble_method == 'drop':
        df_work1 = df_work1.drop_duplicates(cols=cols_1[0])
        df_work2 = df_work2.drop_duplicates(cols=cols_2[0])

    if assemble_method == 'mean':
        df_work1 = df_work1.groupby(cols_1[0]).mean()
        df_work2 = df_work2.groupby(cols_2[0]).mean()

    # Resetting index and combining frames
    df_work1 = df_work1.reset_index()
    df_work2 = df_work2.reset_index()

    # Indexing df_2 to prepare for assemble with df_1
    df_work2 = df_work2.set_index(cols_2[0])

    df_combine = df_work1.join(
        df_work2, on=cols_1[0], lsuffix='_x',
        rsuffix='_y', how=combine_as)

    # Gets rid of empty rows in data frame.
    df_combine = df_combine.dropna()

    # Return data frame to user
    return df_combine


def dataset_add_plate_coordinates_to_frame(df, coordinates):
    """The function adds 3 new columns to inputed data frame.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The parameter called 'df' has input pandas dataframe object.

    coordinates : type(list)
        The parameter called 'coordinates' takes the return value from
        dataset_creating_plate_coordinates_system and adds them
        to df.

    Returns
    -------
    df_work1 : pandas.core.frame.DataFrame
        Returns the 'df' with 3 new columns called
        'Plates', 'Rows' and 'Columns'.

    Raises
    ------
    AssertionError
        if coordinates given is not a list type object.
    """

    # Local globals
    labels = ['Plates', 'Rows', 'Columns']

    # Copying data frame
    df_work1 = dataset_copy_frame(df)

    # Check that coordinates is list type
    try:
        assert type(coordinates) is list

    except AssertionError:
        print ('Parameter called coordinates'
               'is not list type {0}').format(coordinates)

    # Adding new columns to frame
    for i in coordinates:
        df_work1[labels[0]] = coordinates[0]
        df_work1[labels[1]] = coordinates[1]
        df_work1[labels[2]] = coordinates[2]
    # Returning new frame.
    return df_work1


def dataset_creating_plate_coordinates_system(
        df, column_name, pattern='\:|\-'):
    """
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The parameter called 'df' is the input data frame for
        coordinates extraction.

    column_name : type(str)
        The parameter called 'column_name' is strict string type
        entry and it defines where the coordinates are localed
        in data frame.

    pattern : type(str)
        The parameter called 'pattern' contains the regular experssion
        for splitting up the given column into separated entities.

    Returns
    -------
    all_plate_coordinates : list

        The return value called 'all_plate_coordinates' contains
        3 items in the following order plates, rows and columns.

    Raises
    -------
    AssertionError
        If 'column_name' string not found as column name in data frame.

    """

    # Local globals
    plates = []
    rows = []
    columns = []
    rm = []

    # Copying data frame
    df_work1 = dataset_copy_frame(df)

    # Getting column names in data frame

    column_names_df = df_work1.columns.tolist()

    # Testing if columns name exist in data frame
    try:
        assert column_name in column_names_df

    except AssertionError:
        print ("\nThe columns name {0} is not found "
               "in data frame".format(column_name))
        print ("\nPlease enter the following"
               "valid options:\n\t {0}").format(column_names_df)
        return

    # Using pattern to create plate coordinates
    plate_coordinates = [
        re.split(pattern, i) for i in df_work1[column_name].values]

    # Lopping for each column
    for i in plate_coordinates:
        try:
            (int(i[0]))
            (int(i[1]))
            (int(i[2]))

        except ValueError:
            rm.append(i)
    if not rm:
        for i in plate_coordinates:
            plates.append(int(i[0]))
            rows.append(int(i[1]))
            columns.append(int(i[2]))

    # Nesting out rm
    rm = [i[0] for i in rm]

    if len(np.unique(rm)) == 1:
        df_work2 = df_work1[df_work1[column_name] != rm[0]]
        plate_coordinates = [
            re.split(pattern, i) for i in df_work2[column_name].values]
        for i in plate_coordinates:
            plates.append(int(i[0]))
            rows.append(int(i[1]))
            columns.append(int(i[2]))

        # Storing coordinates
        all_plate_coordinates = [plates, rows, columns]

        # Returning all_plate_coordinates
        return all_plate_coordinates, df_work2

    if len(np.unique(rm)) > 1:
        rm = np.unique(rm)
        text = ("Please remove the following values from"
                "column labelled as {0}"
                "values to remove: \n\t{1}").format(column_name, rm)
        print text
        return

    # Storing coordinates
    all_plate_coordinates = [plates, rows, columns]

    # Returning all_plate_coordinates
    return all_plate_coordinates, df_work1


def dataset_plate_position_scoring(data, coordinates):
    """Takes two parameters, "data" is plate matrix of
    32 columns and 48 rows, coordinates are coordinates for
    plate.

    Parameters
    ----------
    data : numpy array
        The parameter called 'data' has shape of 32 columns and 48 rows.

    coordinates : list
        The parameter called 'coordinates' is list with tuples
        for entries. Valid entries range for columns are 0-31 and
        for rows are 0-47.

    Returns
    -------
    data2 : numpy array
        New array where plates coordinates have been scored.

    Raises
    ------
    AsserationError
        If matrix has another shape then 48 rows and 32 columns
        and if coordinates are out of range.

    """
    # Copying numpy array
    data2 = data.copy()
    try:
        assert data2.shape == (32, 48)

    except AssertionError:
        print '\nThe data shape was as not expected 48 rows by 32 columns!'
        return

    # Scoring plate coordinates by adding 1
    for i in coordinates:
        try:
            assert i[0] <= 31 and i[1] <= 47

        except AssertionError:
            print "\nCoordinates out of range."
            return
        data2[i[0]][i[1]] += 1
    return data2


def dataset_bins_calculator(df, binwidth):
    """ Function calculates the proper bin size
    for histogram plot.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        The data frame use for bin size calculations.

    binwidth : float or int (optional)
        The parameter called 'binwidth' determines
        calculations for bin size in histogram.

    Returns
    -------
        columnames, index, edgevalues, bins : list, list, list, int
        The parameter called 'columnames' and 'index' are list
        types returns with string items. The 'edgevalues' variable
        returned is contains the edges for the range of bins
        made. The 'bins' contains the total amount of bins for histogram.

    See Also
    --------
    phen_o_links.dataset_pick_columns : For more information
                                        about "split" parameter.

    """

    # Copying frame
    df_work1 = dataset_copy_frame(df)

    # Picking columns
    columnames, indexnames = dataset_pick_columns(df_work1, split='groupby')

    # Getting edge values.
    edgevalues = [df_work1.min()[columnames].min(),
                  df_work1.max()[columnames].max()]

    # Calculating proper step by edge values given.
    step = ((
        np.ceil(edgevalues[1] + binwidth) -
        (edgevalues[0] - binwidth)) / (2 * binwidth)) * 2

    # Number of bins given by edge value and step.
    bins = np.linspace(edgevalues[0], edgevalues[1] + binwidth, step)
    return columnames, indexnames, edgevalues, bins


def dataset_creating_patches_indices(df_sub, bins, work_columns=[]):
    """ Takes 'bins' return from matplotlib.hist instacemethod and
    calculates the center of the bin.

    Parameters
    ----------

    df_sub : pandas.core.frame.DataFrame(object)
        The parameter called 'df_sub' is pandas data frame object. The
        parameter corresponds 'df_sub' is a sub sampling from a bigger data
        frame.

    bins : numpy.ndarray(object)
        The parameter called 'bins' is the return from matplotlib.hist method
        and its use for calculation for center of bins.

    work_columns : list(object)
        The parameter called 'work_columns' accepts one string entry that
        specifies a label that is present in 'df_sub' to use as length.
        If parameter is left empty function call is triggered.

    Returns
    -------
    df_work1 : pandas.core.frame.DataFrame(object)
        The 'df_work1' has new column called 'patch' with calculation of new
        'bins' centers for figure.

    Raises
    ------
    AssertionError
        If 'work_columns' has more than one value.

    See Also
    --------
    dataset_pick_columns : For more information about function call made if
                           parameter 'work_columns' is left empty.




    """
    # Creating copy frame
    df_work1 = dataset_copy_frame(df_sub)

    if not work_columns:
        print '\nWhat column should be used for patches indices?\n'

        # Pick column to work with.
        work_columns, indexer = dataset_pick_columns(df_work1, split='groupby')

    # Test that only on column is enter
    try:
        assert len(work_columns) == 1

    except AssertionError:
        print ("Function has limit of 1 column"
               ", entry made by user was:\n {0} and"
               "length was {1}").format(work_columns, len(work_columns))
        return

    # Sorting values by given columns in dataframe
    df_work1 = df_work1.sort(columns=work_columns)

    # Getting indices for bar patches.
    df_work1['patches'] = np.searchsorted(
        bins, df_work1[work_columns].values)

    # Returns new data frame
    return df_work1


def dataset_check_duplicated_patches_indices(df_sub):
    """
    Function checks for bar patches duplicates and
    relabels given column set for legend accordingly.

    Parameters
    ----------
    df_sub : pandas.core.frame.DataFrame
        The parameter called 'df_sub' is the return from
        dataset_check_duplicated_patches_indices.

    Returns
    -------
    df_duplicated : pandas.core.frame.DataFrame
        The 'df_duplicated' is the relabelled
        legend column.

    Raises
    ------
    IndexError
        If the variable 'column_picked' is greater than 1

    ValueError
        If data frame lacks the column label 'color' and 'patches'.
    """

    # Rest index for given frame
    df_sub = df_sub.reset_index()

    # Creating copy of data frame
    df_work1 = dataset_copy_frame(df_sub)

    # Prompt text to user
    print ('Pick a column label that contains acceptable values '
           'for legend labelling.')

    # Check that columns contains indices of patches column
    df_columns = df_work1.columns.tolist()
    picked_column, indexer = dataset_pick_columns(df_work1, split='groupby')
    df_checker = ['patches', 'color'] + picked_column

    # Test checker for frame to work

    if not len(picked_column) == 1:
        print ("Please select 1 labelling column,"
               "user picked \n {0} ").format(picked_column)
        raise IndexError
        return

    try:
        [df_columns.index(df_checker[i]) for i in range(len(df_checker))]

    except ValueError:
        print("Column label 'patches','Strain' or 'color' missing"
              "please check for that entry given is"
              "correct: \n {0}").format(df_work1.columns.tolist())
        return

    # Creating indices for corresponding
    first_index = [
        int(i) - 1 for i in df_work1[
            df_work1.patches.duplicated() == True].index]

    second_index = [
        int(i) + 1 for i in df_work1[
            df_work1.patches.duplicated() == True].index]

    duplicated_index = zip(first_index, second_index)

    # Creating 'new' data frame
    df_duplicated = df_work1.copy()

    for i in duplicated_index:
        df_duplicated[picked_column[0]][i[0]:i[1]] = '\n'.join(
            [y for y in df_duplicated[picked_column[0]].values][i[0]:i[1]])
        df_duplicated['color'][i[0]:i[1]] = np.random.choice(
            [y for y in df_duplicated['color'].values][i[0]:i[1]], 1)

    # Rest index
    df_duplicated = df_duplicated.reset_index()

    return df_duplicated, picked_column


def dataset_label_checker(
        df, df2, col_check=[],
        new_column='Column_checker'):
    """Checks for label equality between two data frame columns.

    Parameters
    ----------
    df, df2 : pandas.Dataframe (object)
        Given data frames to check label equality.

    col_check : list (optional)
        The "col_check" takes string labels of columns names found
        in "df" or "df2". Input values are then compared.

    new_column : str (optional)
        The "new_column" parameter is the name for added
        column with boolean values for equality check of labels
    """
    # Copying data frames
    dataframe = dataset_copy_frame(df)
    dataframe2 = dataset_copy_frame(df2)
    try:
        dataframe[col_check[0]]
        dataframe2[col_check[1]]

    except KeyError:
        print "Columns names given are not present in data frame"
        return

    # Logic circuit for compare variable
    compare = len(dataframe) < len(dataframe2)

    if compare:
        # Reverse order for col_check parameter
        col_check = col_check[::-1]

        # Check values for common values in given column.
        label_boolean = dataframe2[col_check[0]].isin(dataframe[col_check[1]])

        # Adds new column with boolean values.
        dataframe2[new_column] = label_boolean

        # Slice of all false values.
        dataframe3 = dataframe2[dataframe2[new_column] == True]

        # Returns modified data frame.
        return dataframe3

    elif compare == False:
        # Check for common values in given column.
        label_boolean = dataframe[col_check[0]].isin(dataframe2[col_check[1]])

        # Adds new column with boolean values.
        dataframe[new_column] = label_boolean

        # Slice of false values.
        dataframe3 = dataframe[dataframe[new_column] == True]

        # Returns modified data frame.
        return dataframe3
    else:
        return 'Ask your local pandas wiz for help!'



def dataset_expected_values(df, median=False, genenorm=[], new_columns=[]):
    """The function return the expected normalised value of experiment.
    Example the observed mean value for XC gene deletion is calculated by
    XC(mean observation) = X(mean observation) + C(mean observation)
    Where X is any gene deletion and C is constant deleted gene
    through out XC mean observation.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        Input data frame.

    median : boolean(optional)
        The parameter called 'median' accepts boolean entries. If boolean
        value is set to True and the calculations are performed with median
        value instead of mean. Default value for median is False.

    genenorm : list
        The constant gene deletion.

    split : str (optional)
        The string parameters used for picking columns.The default value for
        "split" is "groupby".

    new_columns : list (optional)
        The parameter "new_columns" is a list with string
        elements, which corresponds to new column names.

    Returns
    -------
    df_work3 : pandas.DataFrame (object)
        A new data frame with added columns which contains predicted
        double deleted mean observations.

    See Also
    --------
    dataset_copy_frame : For more information about copying data frames.
    dataset_pick_columns : For more information about "split" parameter.
    """
    #Global local
    normgene = []
    # Copying data frame.
    df_work1 = dataset_copy_frame(df)
    # Picking columns to work with.
    wc, idxc = dataset_pick_columns(df, split='groupby')
    # Grouping data frame with given column.
    df_work2 = df_work1.groupby(idxc)[wc]

    # Check flag
    if not median:
        print "mean calc"
        # Getting value for wanted deletion.
        normgene = df_work2.get_group(genenorm[0]).mean()
    if median:
        print "median calc"
        # Getting value for wanted deletion.
        normgene = df_work2.get_group(genenorm[0]).median()
    try:
        # Deleting indexer name from work columns.
        [wc.remove(idxc[i]) for i in range(len(idxc))]

    except ValueError:
        print "Picked columns are accepted"

    if len(new_columns) != len(wc):
        text = ("Columns to added are not the same length as picked columns to"
                " work with new columns length is {0} and columns to work "
                "with had a length {1}").format(len(new_columns), len(wc))
        raise ValueError(text)

    # Looping given names for new column.
    for i in range(len(new_columns)):
        df_work1[new_columns[i]] = df_work1[wc[i]].values + normgene[wc[i]]

    # Setting index to data frame.
    df_work1 = df_work1.set_index(idxc[0])

    for i in new_columns:
        df_work1.set_value(str(genenorm[0]), str(i), 0)

    df_work1 = df_work1.reset_index()

    # Returning data frame.
    return df_work1


def dataset_regline(x_data, y_data):
    """ The function returns regression parameters as slope, intercept,
    p-value etc.

    Parameters
    ----------
    x_data, y_data : numpy.array like sequences
        Parameters must be a sequence of digits and have the same
        length and dimension.

    Returns
    -------
    slope1 : float
        The slope calculated from a linear function of data input.

    intercept1 : float
        The intercept from the linear function of data input.

    r_value1 : float
        The correlation coefficient from data input.

    p_value1 : float
        Two tailed p-value with null hypothesise of slope being zero

    std_err1 : float
        Standard error of the estimate

    Raises
    ------
    ValueError
        If input data is not the same dimension and length and if
        resulting slope returns NAN.

    See Also
    --------
    Regression line :
        For more information regarding
        regression line calculations see
        scipy documentation for stats.linregress.

    """
    if x_data.shape != y_data.shape:
        raise ValueError("Dimension and lengths"
                         "for x and y input do not match")
        return
    # Statistical facts
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
        x_data, y_data)
    try:
        slope1.as_integer_ratio()
    except ValueError:
        print "Either x or y sequence contains NAN values"
        return
    return slope1, intercept1, r_value1, p_value1, std_err1


def dataset_filter_within_a_range_of_value(df, filter_val, split="groupby"):
    """Takes a pandas data frame object and returns picked columns filtered by
    filter value. The test for respective columns is: x '<=' a and -a '>=' x,
    where 'a' is the filter value and 'x' is the column values.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        Data frame input.

    filter_val : float, int
        Thresh value for picked columns to not be equal or greater
        than set "filter_val".

    split : str (optional)
        Parameter used in dataset_pick_columns
        and default value is "groupby".

    Returns
    -------
    df_work2 : Data frame pandas object
        Returns 'df' input, with a new column called as above.
        the new column contains boolean for values that passes
        the given test.

    df_work3 : Data frame pandas object
        Returns a subset from 'df' input.

    x, y : numpy array like
        The column values for filtered columns.

    Raises
    ------
    ValueError
        if "filter_val" is not a digit of any kind.

    See Also
    --------
    data_set.dataset_pick_columns : For more information regarding "split"
                                    parameter.
    """
    try:
        float(filter_val)
    except ValueError:
        print "Filter value entered is not a digit please try again!"
        return
    # Global local
    val = filter_val
    new_name = 'Values_between_' + str(val) + '_and_' + str(np.negative(val))
    # Creating copy of data frame
    df_work1 = dataset_copy_frame(df)

    # Prompts to user!
    print ("\nCreate a subset from data frame.\n"
           " By following the instructions:\n\n")

    # Picking columns to work with
    column_names, grouper = dataset_pick_columns(df_work1, split)

    # Re ordering picked columns
    column_names = grouper + column_names

    # Creating subset of pick columns
    df_work2 = df_work1[column_names]

    print "\n Pick the x and y values for filtering subset. \n"

    # Picking columns to filter
    column_names2, grouper2 = dataset_pick_columns(df_work2, split)

    all_columns = grouper2 + column_names2

    # Testing conditions for values in data frame.
    c1 = (df_work2[column_names2[0]] <= filter_val) & (df_work2[
        column_names2[1]] <= filter_val)

    c2 = (filter_val * -1 <= df_work2[column_names2[0]]) & (
        filter_val * -1 <= df_work2[column_names2[1]])
    print "work"
    # Adding conditions to frame.
    df_work2["Not_over"] = c1
    df_work2["Not_under"] = c2
    df_work2[new_name] = (
        df_work2["Not_over"] == True) & (df_work2["Not_under"] == True)

    # Sub sampling
    df_work3 = df_work2[df_work2[new_name] == True]

    x = df_work3[column_names2[0]]
    y = df_work3[column_names2[1]]

    # Getting columns names.
    columns = df_work2.columns.tolist()
    [columns.pop(columns.index(i)) for i in all_columns]

    # Removing columns that are unnecessary.
    df_work2 = df_work2.drop(columns[:-1], axis=1)
    df_work3 = df_work3.drop(columns, axis=1)

    return df_work2, df_work3, x, y, all_columns


def dataset_copy_frame(df):
    """Takes a copy of pandas Data frame object and returns it

    Parameters
    -----------
    df : pandas.DataFrame
        Input data frame

    Returns
    --------
    df_work : pandas.DataFrame
        Creates a copy of input data frame
    """
    # Copying data frame
    df_work = df.copy()

    # Returns copy of dataframe
    return df_work


def dataset_clear_columns(df, pattern='Unnamed:\s\d'):
    """Gets rid of unwanted column(s) with specified regular
    expression pattern.

    Parameters
    ----------
    df : pandas.DataFrame

    pattern : str, optional
        Description of parameter 'pattern' (the default values is
        'Unnamed:<backslash>s<backslash>d', formated as regular expression)

    Returns
    -------
    df2 : pandas.DataFrame
        The data frame without unwanted column specified by pattern parameters.
    """

    # Copying data frame
    df_work1 = dataset_copy_frame(df)

    try:
        df_work1.select(lambda x: not re.search('Unnamed:\s\d', x), axis=1)

    except TypeError:
        print "Empty column(s) not found in data set."
        return df_work1

    # Removes unwanted columns
    df2 = df_work1.select(lambda x: not re.search('Unnamed:\s\d', x), axis=1)

    # Returns data frame
    return df2


def dataset_filesave(
    df, filename='', delimiter='\t',
    header=1, filesuffix='.csv',
        pathdir='./'):
    """Saves a pandas data frame as tab separated vector file or tsv-file.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame

    filename : str
        File name given to file.

    delimiter : str, optional
        The description of 'delimiter' the character
        the formats the file structure
        (the default value for 'delimiter' is <TAB>).

    header : int, optional
        The 'header' parameter determines what row is seen
        as column names (the default value for 'header' is 1).

    filesuffix : str, optional
        The file suffix (the default value for 'filesuffix' is .csv).

    pathdir : str, optional
        'pathdir' locates the output file to
        a directory (the default value for pathdir is './').

    Returns
    -------
        Save file prompt and absolute path to file

    Raises
    ------
    OSError
        Warns user of path to directory not found!

    See Also
    --------
    dataset_copy_frame : For more information about "df_work1".
    """
    # Raise exception
    try:
        listdir(str(pathdir))
    except OSError:
        print 'path for directory not found!'
        return 'Please try again'

    # Copy data frame.
    df_work1 = df.copy()

    # Creating list of files found by pathdir.
    pathcwd = listdir(pathdir)

    # Variable count for logic circuit.
    count = sum([pathcwd[i] == filename + filesuffix for i in range(
        len(pathcwd))])
    if count < 1:
        df_work1.to_csv(
            pathdir + filename + filesuffix, sep=delimiter, header=header)
        print "Your file is located here:"
        print abspath(pathdir + filename + filesuffix)

    else:
        add_suffix = str(len(pathcwd) + 1)
        df_work1.to_csv(
            pathdir + filename + add_suffix + filesuffix,
            sep=delimiter, header=header)
        print "Your file is located here:"
        print abspath(pathdir + filename + add_suffix + filesuffix)


def dataset_pick_columns(df, split='', pattern='Unnamed:\s\d'):
    """Takes a pandas data frame object and returns
    picked columns. Function has a split parameters,
    which returns either a list of columns names
    or a new data frame and the indexer column name.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame object from pandas.

    split : str, optional
        Description of parameter 'split'(The accepted string values are
        'indexby' or 'groupby'.)

    Returns
    -------
    df2, col_names : pandas.DataFrame, list
        When split is 'indexby'.

    new_cols, new_index : list
        When split is 'groupby'.

    Raises
    ------
    TypeError
        If input digit lacks a comma and can't be regarded as list type.

    See Also
    --------
    dataset_clear_columns : For more information about 'pattern' parameter.
    """

    # Get rid of useless column
    df = dataset_clear_columns(df, pattern)

    # List columns
    cols = df.columns.tolist()

    # Converts list to dictionary
    dict_cols = {i: cols[i] for i in range(len(cols))}

    # Prompts to screen
    print "Columns found in data set:\n"
    pprint(dict_cols)

    # Input container
    picked = input('Pick columns to work with via the numbers'
                   ' e.g. 1,2,3 etc\n\t: ')
    picked_index = input('Pick a indexer or groupby for data frame here \n\t:')

    try:
        list(picked)
        list(picked_index)
    except TypeError:
        print 'Please add comma after digit e.g "0,"'
        return
    try:
        cols[picked[-1]]
    except IndexError:
        print '\nPlease pick existing columns from Data frame'
        return

    # List input container
    cols_picked = list(picked)
    index_picked = list(picked_index)

    # Returns values form container
    new_cols = [dict_cols.get(cols_picked[i]) for i in range(len(cols_picked))]
    new_index = [dict_cols.get(index_picked[i])
                 for i in range(len(index_picked))]

    # Creates new_data frame
    if split == 'indexby':
        df2 = df[new_cols]
        df2 = df2.set_index(new_index)
        col_names = new_index

        # Returns data frame and its columns
        return df2, col_names

    elif split == 'groupby':

        # Returns data frame and its columns
        return new_cols, new_index

    else:
        return 'Error'


def dataset_flag_remover(
        df, pattern='\Y\w*\S\S', split='groupby', flag1='Q:WTS:',
        flag2='Q:H0S:', mark1='REF', mark2='EXP'):
    """Removes a give flag for a specified column with dimension of 1x1 from
    a pandas data frame object and returns given marker.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame

    pattern : str(optional)
        The parameter called 'pattern' takes a given regular expression
        for the re module and search for its in the given column. The
        type of 'pattern' entered should be a sting.

    flag1, flag2 : str, optional
        String removed as str.lstrip()
        Description of  of parameters 'flag1', 'flag2' (The default value of
        flag1 is 'Q:WTS:', and the value of flag2 is 'Q:H0S:')

    mark1,  mark2 : str, optional
        Marker for genetic background.
        Description of parameter 'mark1', 'mark2' (The default value are
        mark1 is 'REF' and mark2 is 'EXP')

    Returns
    -------
    df_work1, geneback : pandas.DataFrame, str
        Returns data frame without given flags and with the given sting label.

    See Also
    --------
    dataset_pick_columns : For more information about 'split' parameter.
    dataset_copy_frame : For more information about copying data frames.

    Notes
    -----
    """
    # Local global
    search_hits = []
    # Copying data frame
    df_work1 = dataset_copy_frame(df)

    # Picking columns
    columns_work, grouper = dataset_pick_columns(df_work1, split)

    # Slice column for pattern search
    txt = df_work1[grouper[0]]

    # Takes away white spaces and wraps str
    txt2 = txt.str.replace(" ", "")

    # Check for common pattern in csv-file
    var1 = txt2.str.contains(flag1)
    var2 = txt2.str.contains(flag2)

    # Some logic that separates reference from altered strain
    if list(var1)[0]:

        # Returns background
        geneback = mark1

    if list(var2)[0]:
        # Returns background
        geneback = mark2

    for i in df_work1[grouper].values:
        hits = re.search(pattern, str(i))
        if hits:
            search_hits.append(hits.group())
        if not hits:
            search_hits.append('Not found')

    df_work1[grouper] = search_hits

    # Returning Dataframe
    return df_work1, geneback


def dataset_groupby_triplicate(df, key_column=['']):
    """ Takes a pandas data frame and sections it
    by a given column. The column is assumed to contain
    a max of 3 indices per label.

    A group is defined by containing:
        forms groups of 3 = indices that ==3 or >3
        forms groups of 2 = indices that are <3 but >1
        forms groups of 1 = indices that have at most 1

    The indices are given in 'key_column'.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The give input data frame to be sectioned.

    key_column : list(str(optional))
        The parameter called 'key_column' accepts a string value present as a
        column name in 'df'. The order of input affects function behavior, the
        first element is the main label and the following labels are subs
        sequential divisors for data.

    Returns
    -------
    groups : numpy.array_like
        The 'groups' return is the indices found in 'key_column'
        split or section by above algorithm.

    """

    # Local Global
    groups = []

    # Copy frame
    df_work1 = dataset_copy_frame(df)

    # Sort by column
    df_work1 = df_work1.sort(columns=key_column)

    # Grouping data frame by column
    grouped = df_work1.groupby(key_column)

    # Super intensive memory draining for loop
    for k, labels in grouped[key_column]:
        i = labels[key_column[0]].values
        if len(i) == 3:
            groups.append(i)
        if len(i) == 2:
            groups.append(i)
        if len(i) == 1:
            groups.append(i)
        if len(i) > 3:
            if not len(i) % 3:
                n = len(i) / 3
                split = np.array_split(i, n)
                for nr in range(len(split)):
                    groups.append(split[nr])
            if len(i) % 3 == 2:
                array = i[:-2]
                remain = i[-2:]
                n = len(array) / 3
                split2 = np.array_split(array, n)
                for nr2 in range(len(split2)):
                    groups.append(split2[nr2])
                groups.append(remain)
            if len(i) % 3 == 1 and len(i) > 2:
                array = i[:-1]
                remain = i[-1:]
                n = len(array) / 3
                split3 = np.array_split(array, n)
                for nr3 in range(len(split3)):
                    groups.append(split3[nr3])
                groups.append(remain)
    return groups


def dataset_add_groups(df, label_column=['']):
    """ Function sections/divides a data frame by 3, a given column present
    in given data frame.
    Adds to new column names to given data frame
    'Groups' = is the 'label_column' values + integer.
    'Counts' = is the number of time indices a group contain in total.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input data frame to be divided.

    label_column : list(str(optional))
        The 'label_column' accepts string typed values.
        The string entered must be present in 'df'
        to work.

    Returns
    -------
    df_work3 : pandas.core.frame.DataFrame(object)
        The new frame with additional columns and sections.

    Saves 2 tab delimited csv-files called
        file nr 1 is called 'table_grouped_by_indices'
        file nr 2 is called 'table_grouped_by_indices_with_counts_numbers_one'

    File nr 2 are values that are present by one or less indices.

    See Also
    --------
    dataset_groupby_triplicate : For more information on how division is made
    dataset_filesave : For more information on csv-file created.

    """
    # Local global
    groups = dataset_groupby_triplicate(df, key_column=label_column)
    labels_nr = []
    filenamed = 'table_grouped_by_indices'
    filenamed2 = 'table_grouped_by_indices_with_counts_numbers_one'

    # Copy frame
    df_work1 = dataset_copy_frame(df)

    # Sort frame by column
    df_work1 = df_work1.sort(columns=label_column)

    # Adding groups to labels
    for i in range(len(groups)):
        for label in groups[i]:
            labels_nr.append(str(label) + '_' + str(i))

    # Inserting new column
    df_work1.insert(0, 'Groups', labels_nr, allow_duplicates=True)

    # Filtering labels by counts of indices.
    indices = df_work1.Groups.value_counts()
    indices = indices.reset_index()
    indices.columns = ['Groups', 'Counts']

    # Creating dictionary of indices.
    indices_dict = dataset_to_dict(indices, keys='Groups', values='Counts')

    # Adding dictionary to new frame
    df_work2 = dataset_add_column_by_dict(
        df_work1, indices_dict, Grouper='Groups', new_col='Counts')
    # Filtering by counts columns
    df_work3 = df_work2[df_work2.Counts != 1]
    df_work4 = df_work2[df_work2.Counts == 1]

    dataset_filesave(df_work3, filename=filenamed)
    dataset_filesave(df_work4, filename=filenamed2)

    # Returns frames with indices for Counts values > 2.
    return df_work3


def dataset_reflag(
        df, geneback, check1='R', check2='E',
        name1='Genetic Background', name2='Genetic Background',
        name1column=0, name2column=0, filename='reflagged',
        add_suffix1='_ref', add_suffix2='_exp'):
    """Flags a pandas data frame object for given values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame pandas.

    geneback : str
        dataset_flag_remover : Inherent string value.

    check1, check2 : str, optional
        Flag read as first letter from 'geneback' (The default string values
        for check1 is 'R', and for check2 is 'E').

    name1, name2 : str, optional
        Name for new column to be (The default value for
        'name1' and 'name2' is 'Genetic Background' ).

    add_suffix1, add_suffix2 : str, optional
        Filename suffix add to 'filename' parameter (The default value
        for 'add_suffix1' is '_ref' and for add_suffix2 is '_exp').

    Returns
    -------
    df_work2 : pandas.DataFrame
        Returns new data frame with new column and flag.

    See Also
    --------
        dataset_filesave : For more information about 'filename' parameter.
        dataset_flag_remover : For more information about 'geneback' parameter.
    """
    # Copying data frame
    df_work1 = dataset_copy_frame(df)

    # Get rid of unwanted column
    df_work2 = dataset_clear_columns(df_work1)

    # Logic to support above statement
    if geneback.startswith(check1):
        df_work2.insert(name1column, name1, value=str(geneback))
        filename = filename + add_suffix1
        dataset_filesave(df_work2, filename)
        return df_work2
    elif geneback.startswith(check2):
        filename = filename + add_suffix2
        df_work2.insert(name2column, name2, value=str(geneback))
        dataset_filesave(df_work2, filename)
        return df_work2
    else:
        print ("Dataset was not re-flagged."
               "Check if parameters have the proper values ")
        return df_work2


def dataset_to_dict(df, keys='', values=''):
    """Creates a dictionary of of two columns within a data frame and
    returns a dataset dictionary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    keys : str, optional
        Column name with key values for dictionary

    values : str, optional
        Column name with item values for dictionary

    Returns
    -------
    dataset_dict : dictionary

    Raises
    ------
    KeyError
        If string either string is not found in column name

    See Also
    --------
    dataset_copy_frame : For more information about data frame copy.
    """
    # Copying data frame
    df_work1 = dataset_copy_frame(df)
    try:
        df_work1[[keys, values]]

    except KeyError:
        print ("Column names {0}, {1} does "
               "not exist in data frame".format(keys, values))
        return

    # Creating dictionary
    dataset_dict = df_work1.set_index(keys)[values].to_dict()

    # Returns
    return dataset_dict


def dataset_dict_to_dataset(keys=[], value=''):
    """ Creates dictionary object that can be added to pandas.DataFrame(object)
    if column values are equivalent to values found in 'keys'.

    Parameters
    ----------
    keys : list(object)
        The parameter called 'keys' is list object that only accepts
        nested items.

    value : str(object)
        The parameter called 'value' is str object that only accepts
        string as input.

    Returns
    -------
    new_dict : dict(object)
        The return is called new_dict and contains a dictionary where 'keys'
        are the keys for the dictionary and where value has been added with a
        unique index.

    Raises
    ------
    ValueError
        If either 'keys' or 'value' are not formatted properly or
        that input if wrong type.
    """

    # Global Local
    new_dict = {}

    # Check for user input
    if not(all(isinstance(i, list) for i in keys)):
        text = ("The list object called 'keys' only accepts nested items"
                " see below:\n\n [[1, 2, 3], ['A'], [2, 'Z']]")
        raise ValueError(text)

    if not(all(isinstance(i, str) for i in value)):
        text2 = ("The string object called 'value' only accepts string input"
                 " see below:\n\n value = 'Hola'")
        raise ValueError(text2)

    # Creating dictionary for adding to pandas.Dataframe(object)
    for i, group in enumerate(keys):
        for key in group:
            new_dict.update({key: value + str(i)})
    return new_dict


def dataset_add_column_by_dict(df, dataset_dict, Grouper='', new_col=''):
    """The function adds a new column to a given pandas data frame.
    The addition column is added because the keys for each value from
    dataset_dict are found within the data frame
    via the Grouper which is column name.

    Parameters
    ----------
    df : pandas.DataFrame
        Inserted data frame to add column.

    Grouper : str, optional
        "Grouper" the labelling.

    dataset_dict : dictionary
        'dataset_dict' is the obtain dictionary from a pandas.DataFrame

    new_col : str, optional
        String label for new column. (The default value is empty.)

    Returns
    -------
    df_work1 : pandas.DataFrame
        New data frame with a new column

    Raises
    ------
    AttributeError
        If dataset_dict is not a dictionary type.

    See Also
    --------
    dataset_copy_frame : For more information regarding data frame copies.
    dataset_pvalue : For more information regarding
                     parameters "Grouper", "p_value_dict"
    """
    # Copying data frame
    df_work1 = dataset_copy_frame(df)

    # Creating np.array from Grouper
    keys = df_work1[Grouper].values

    # Issue raiser
    try:
        dataset_dict.viewvalues()

    except AttributeError:
        print "dataset_dict is not of type dict"
        return

    # Adding new column.
    df_work1[new_col] = [dataset_dict.get(i, i) for i in keys]

    # Returns data frame with new column.
    return df_work1


def dataset_SEM_calculator(df, value_columns=[], counts=[]):
    """The function returns the Standard Error of the Mean(SEM) of
    any given data frame that contains 2 columns names at least.
    Names with standard deviation (std) should be flagged with
    following '_std' at the end of each column name.

    if 'value_columns' or 'counts' are empty user if force
    to pick out columns.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input frame with wanted values.

    value_columns : list(object)
        Parameter called 'value_columns' items are string object which
        corresponds to a column name.

    counts : list(object)
        Parameter called 'counts' accept a string a present
        column name in df. The 'counts' takes 1 item in calculation.

    Returns
    -------
    df_work1 : pandas.core.frame.DataFrame(object)
        The df data frame with new sets of columns flagged
        '_SEM' containing SEM values.

    Raises
    -------
    ZeroDivisionError
        If input 'df' lacks column names with flag '_std'
        for standard deviation columns.

    """
    # Local Global
    values_SEM = []
    new_column = []

    # Copying frame
    df_work1 = dataset_copy_frame(df)

    # Logic circuit
    if not (value_columns and counts):
        value_columns, counts = dataset_pick_columns(df_work1, split='groupby')
    try:
        1.0 / len([i for i in df_work1.columns if i.endswith('_std') == True])
        1.0 / len([i for i in value_columns if i.endswith('_std') == True])

    except ZeroDivisionError:
        print ("No column names with standard deviation flagged as '_std'"
               " was found please flag column names with the suffix '_std'"
               " for function to work!!")
        return

    # Rendering new names
    new_column = [i.replace('_std', '_SEM') for i in value_columns]

    # Looping and Calculating SEM for respective column
    for i in value_columns:
        SEM = df_work1[i].values / np.sqrt(df_work1[counts[0]].values)
        values_SEM.append(SEM)

    # Adding Column to data frame
    for i in range(len(new_column)):
        df_work1[new_column[i]] = values_SEM[i]

    # Returning Frame
    return df_work1


def dataset_orfnames_to_genenames(df, keys=[], values=[]):
    """ Takes any given pandas data frame and adds 1 column only
    called 'Gene', which are equal to standard names for present
    orf in data frame.
    """

    # Local Global
    GO_terms = dataset_import_goterms(go_filter=['P', 'C', 'F'])
    df_work1 = dataset_copy_frame(df)
    columns = df_work1.columns.tolist()

    # Check that keys and values are empty.
    if not (keys and values):

        print ("\n Remember that columns picked have the same function as"
               "\n python dictionary {keys : values}\n"
               "\n Pick 1 work column (value) and 1 index column (key).\n\n")

        # Picking columns
        values, keys = dataset_pick_columns(GO_terms, split='groupby')

    # Creating dictionary
    orf_to_genes = dataset_to_dict(GO_terms, keys=keys[0], values=values[0])

    try:
        columns.index(keys[0])

    except ValueError:
        print (" Value for 'keys' not present in 'df'"
               " Columns present in 'df' are : \n {0} \n"
               " And 'keys' value picked is {1}").format(columns, keys[0])
        return

    # Adding 1 column to 'df'
    df_work2 = dataset_add_column_by_dict(
        df_work1, orf_to_genes, Grouper=keys[0], new_col=values[0])

    df_work2 = df_work2.reset_index()

    # Compares gene names and systematic.
    for i in df_work2.Gene.index:
        if df_work2.Gene.isnull()[i]:
            df_work2.Gene.values[i] = df_work2.ORF.values[i]

    return df_work2


def dataset_create_coordinate_system(colony_format):
    """ Constructs plates coordinates with the layout as 8 rows and 12 columns.

    Parameters
    ----------
    colony_format : int, float(object)
        The 'colony_format' is a integer that contains a 96
        and multiplied by constant (c).
        colony_format = 96 * c

    Returns
    -------
    coordinates
        A numpy.array object.

    Raises
    ------
    ValueError
        If 'colony_format' is not divisible by 96 and if
        layout dimensions differ from 8 X 12 format.
        If 'colony_format' cannot be converted to integer.
    """
    # Global Local
    format_96 = int(colony_format) % 96
    rows = 8
    columns = 12

    # Logic circuit to determine if it's divisible by 96.
    if not format_96:

        # Calculates scalar
        scalar = int(colony_format / 96) ** 0.5

        # Check that layout corresponds to (8,12)
        if scalar > 1 and not scalar % 2:
            rows = rows * scalar
            columns = columns * scalar

            # Getting coordinates
            coordinates = np.arange(colony_format).reshape(rows, columns)
            return coordinates

        # Exception for scalar equal to 1.
        if scalar == 1:
            coordinates = np.arange(colony_format).reshape(rows, columns)
            return coordinates
        else:
            text = ("\nThe parameter 'colony_format' cannot be shaped "
                    " as: \n 8 X 12 \n layout.\nPlease make sure of that!")
            raise ValueError(text)
    else:
        text2 = ("\nThe value given to 'colony_format'"
                 " cannot be factored as:\n"
                 " 'colony_format' = 96 x scalar.")
        raise ValueError(text2)


def dataset_get_plate_peripheral(plate_coordinates):
    """ Takes return output from data_set.dataset_create_coordinate_system and
    returns a pandas.core.frame.DataFrame(object) with border coordinates.

    Parameters
    ----------
    plate_coordinates : numpy.array(object)
        A numpy array object with values.

    Returns
    -------
    plate_borders
        A pandas DataFrame object with 2 columns 'R' for row and 'C' for
        column.

    Raises
    ------
    ValueError
        If parameter 'plate_coordinates' is not a numpy array object.
    """
    # Global Local
    borders = []

    isnumpy = all(isinstance(i, type(np.array(0))) for i in plate_coordinates)

    if not isnumpy:
        text = ("\nThe parameter given as 'plate_coordinates' is not a\n"
                " numpy array object. Please try input only numpy arrays")
        raise ValueError(text)

    # Creating rows and columns coordinates
    columns_to_remove = np.concatenate((
        plate_coordinates[0], plate_coordinates[-1]))
    rows_to_remove = np.concatenate((
        plate_coordinates.T[0], plate_coordinates.T[-1]))
    # Loop to append wanted removable values
    for i in columns_to_remove:
        borders.append(np.ravel(np.where(plate_coordinates == i)))
    for i in rows_to_remove:
        borders.append(np.ravel(np.where(plate_coordinates == i)))

    # Creating pandas.Dataframe(object)
    plate_borders = pd.DataFrame(borders, columns=['R', 'C'])

    # Dropping duplicated values
    plate_borders = plate_borders.drop_duplicates()

    # Return data frame
    return plate_borders


def dataset_withoutborders(df, borders):
    """ Takes a given subset with only border colonies 'His3' in
    SGA collection.

    Parameters
    ----------
    df : pandas.DataFrame(object)
        The sub set containing only border colonies.
    borders : pandas.DataFrame(object)
        Contains border colonies.

    Returns
    -------
    df_truth : pandas.core.frame.DataFrame(object)
        A pandas data frame with only values outside of
        border definition

    Raises
    ------
    ValueError
        If 'Coordinates' values given as border for data set
        in 'df' returns boolean 'False'. For position (0,0) and (0,47)
    """
    # Copy frames.
    df_work1 = df.copy()
    borders_1 = borders.copy()

    # Creating coordinates
    df_work1['Coordinates'] = zip(df_work1['C'].values, df_work1['R'].values)
    borders_1['Coordinates'] = zip(
        borders_1['R'].values, borders_1['C'].values)

    # Checking cross exam values
    df_work1['Borders'] = df_work1.Coordinates.isin(borders_1.Coordinates)

    # Check that array is viable
    test_xy = np.ravel(np.where(
        df_work1[df_work1.Coordinates.values == (
            '(0.0, 47.0)')]['Borders'].values == False))

    test_xy1 = np.ravel(np.where(
        df_work1[df_work1.Coordinates.values == (
            '(0.0, 0.0)')]['Borders'].values == False))

    if not(test_xy or test_xy1):

        # Data save file
        dataset_filesave(df_work1, filename='dataset_without_marked_borders')

        # Setting filtering for True.
        df_truth = df_work1[df_work1.Borders == False]

        # Return
        return df_truth
    else:
        text = ("\n The Coordinates given are in the wrong format."
                "\n Please retry with a different layout for plates."
                "\n Coordinates (0,0) or (0,47) has returned 'False'. ")

        raise ValueError(text)


def dataset_median_batch_effect_calc(df,columns=[]):
    """ The function calculates a batch effect via the obtained median values.
    The calculation is based by pair-wise comparison within a group
    and groups are calculated as follow, nr_groups = (n*(n-1)) / 2.
    """
    # Local global
    new_labels=[]
    # Data frame copy
    df_work1 = dataset_copy_frame(df)

    # Check for columns not empty
    if not any(columns):
        columns, index = dataset_pick_columns(df_work1, split='groupby')

    # Check that columns parameter is the right size!
    try:
        if not len(columns) == 2:
            raise ValueError
    except ValueError:
        text = "Parameter called 'columns' more than two items in it:\n"
        print text, columns
        return
    # Sub dividing data frame to median and labels
    df_sub = df_work1[columns]

    # Creating array
    medians = df_sub[columns[-1]].values

    # Creating abs difference for all median values
    sub_median = np.abs(medians[:, None] - medians)

    # Removing indices that contain zeros!
    n_medians = medians.size

    rm_idx = np.arange(n_medians) * (n_medians + 1)

    # Slice of unwanted values
    abs_medians = np.delete(sub_median, rm_idx)

    abs_medians = abs_medians[:int(len(abs_medians) * 0.5)]

    # Creating new labels!
    labels = df_work1[columns[0]].values

    new_labels = [str(i) + '_vs_' + str(j)
                  for i in labels for j in labels if i != j]
    new_labels = new_labels[:int(len(new_labels) * 0.5)]

    print new_labels
    print abs_medians

    # Creating new data frame
    df_median = pd.DataFrame(
        abs_medians, new_labels, columns=['Absoulte_diff_Median'])

    # New and old frame returned
    return df_work1, df_median


# Main worker functions

def dataset_export_ORF_names_for_GO_Enrichment(
        df, column='', suffix='',
        separator="\n", cwd=''):
    """Function writes a text file that is delimited by
    new line. The file data is written from items in
    'column' and file created at working directory.

    Filename is set to 'GO_enrichment_query_from' + column + suffix.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The data frame with wanted query names for GO-enrichment analysis.

    column : str(optional)
        The column name where query names are a taken

    suffix : str(optional)
        The file suffix can be any valid text file format.

    separator : str(optional)
        The 'separator' determines how items should be delimited
        in text-file.

    cwd : str(optional)
        The 'cwd' is set to './', which is current working directory.

    Returns
    -------
    Writes a text-file in current working directory.

    Raises
    ------
    ValueError
        If 'column' string not found in column names of input data frame.
    IOError
        If 'filename' not found in system.
    """
    # Local global
    columns = df.columns.tolist()
    filename = 'GO_enrichment_query_from'

    # Copy frame
    df_work1 = dataset_copy_frame(df)

    # Logic circuit
    if not suffix:
        suffix = '.txt'
    if not cwd:
        cwd = "./"
    try:
        columns.index(column)
    except ValueError:
        print (" Column named \n\t{0}\n"
               "Not found in \n\n{1}\n\n").format(column, columns)
        return
    filename = cwd + filename + '_' + column + suffix
    print 'here is before altering:\n', filename
    if exists(filename):
        filenr = '_' + str(len(listdir(cwd)) + 1)
        filename = filename[
            len(cwd):filename.index('.', len(cwd))] + filenr + filename[
                filename.index('.', len(cwd)):]
        print 'after altering ', filename

    df_work1[column].unique().tofile(filename, sep=separator)
    print 'filename outside logic box ', filename

    if exists(filename):
        print ("Your file is located at :\n\t{0}").format(abspath(filename))
        return
    if not exists(filename):
        print ("Something is wrong with filename"
               " please check that either directory"
               " path exist or that file exist. ")
        raise IOError
        return


def dataset_import_goterms(
        delimiter="\t", go_filter=["P"], filepath="./sgd_go_slimterm.csv"):
    """Import csv-file with GO slim terms and filters by GO aspects. The
    columns are renamed 'ORF' for systematic nomenclature loci used by SGD,
    'Gene' as the standard gene name, 'SGDID' for the unique identifier used by
    SGD database, 'GO_Aspect' for the type ontology, 'GO_Slim_Term' for the
    GO annotation covered by SGD, 'GOID' for the unique numerical pointer of a
    GO-term used by SGD and lastly the 'Feature_type' which specifies the type
    of e.g. Dubious or tRNA.

    Parameters
    ----------
    delimiter : str(optional)
        The parameter delimiter specifies the delimiter for the file import.

    go_filter : str(optional)
        The parameter 'go_filter' can be set to different GO Slim Term stated
        ontology gained from SGD. Accepted entries is 'P' for biological
        process, 'F' for molecular function and 'C' as cellular component. The
        parameter 'go_filter' is set to 'P' as default.

    filepath : str(optional)
        A relative path to a csv-file containing GO-annotation from current
        working directory. The default value was set to './sgd_go_slimterm.csv'

    Returns
    -------
    t_GO : pandas.core.frame.DataFrame(object)
        The 't_GO' contains the filtered GO Slim Term from SGD. The 't_GO' can
        be then added to a new pandas.core.frame.DataFrame(object)

    Raises
    ------
    AssertionError
        If the relative path to the csv-file (with GO-annotations) given by
        the parameter called 'filepath' does not exist.

    AssertionError
        If column count exceeds or is less than 7 for the given GO annotation
        file.

    ValueError
        If relabelled columns are not valid.

    See Also
    --------
    dataset_clear_columns: For more information about function.

    dataset_add_sgd_goslimterms : For more information about how to add
                                  GO-terms to a another pandas DataFrame
                                  (object).

    Notes
    -----
    The 'go_filter' entries are the sames as the once find at the given web
    page below for the file called 'go_slim_mapping.tab'. The file was renamed
    from  'go_slim_mapping.tab' to 'sgd_go_slimterm.csv'. For more information
    please visit: http://downloads.yeastgenome.org/curation/literature/

    """
    # Check for existence of file.
    try:
        assert(exists(filepath))

    except AssertionError:
        text = ("The file that was given a path to by 'filepath' does not "
                "exist! The user input was:\n {0}").format(str(filepath))
        print text
        return

    # Importing file.
    t_none_formatted = pd.read_csv(filepath, delimiter=delimiter, header=None)

    # Clear all empty columns.
    t_format = t_none_formatted.copy()

    # Check that imported file has 7 columns.
    try:
        assert(t_format.columns.shape[0] == 7)

    except AssertionError:
        text = ("Imported file named: {0} contains more or less then 7 "
                "columns. Columns found in imported file called {0} was:\n "
                "{1} .").format(filepath, len(t_format.columns))
        print text
        return t_format

    # Adding column names to variable.
    column_names = ['ORF', 'Gene', 'SGDID', 'GO_Aspect', 'GO_Slim_Term',
                    'GOID', 'Feature_type']

    column_default = t_format.columns.tolist()

    # Renaming columns to 'column_names'.
    t_named = t_format.rename(
        columns={column_default[i]: column_names[i] for i in range(
            len(column_default))})

    # Check that names are valid.
    try:
        t_named.columns.tolist().index('Gene')
    except ValueError:
        text = ("Renaming of columns may have failed please check that "
                "columns have be relabelled as {0} .").format(column_names)
        print text

    # Compares gene names and systematic.
    for i in t_named.Gene.index:
        if t_named.Gene.isnull()[i]:
            t_named.Gene.values[i] = t_named.ORF.values[i]

    # Getting rid of GOIDs equal to NaN-values.
    for i in t_named.GOID.index:
        if t_named.GOID.isnull()[i]:
            t_named = t_named.drop(i)

    # Filter data frame by go_filter
    if not go_filter or len(go_filter) > 2:
        return t_named
    if len(go_filter) == 1:
        t_GO = t_named[t_named['GO_Aspect'] == go_filter[0]]

        # Return data frame with filtered GO slim terms.
        return t_GO

    if len(go_filter) == 2:
        t_GO = t_named.loc[(t_named['GO_Aspect'] == go_filter[0]) | (
            t_named['GO_Aspect'] == go_filter[1])]

        # Return data frame with filtered GO slim terms.
        return t_GO
    print 'end'


def dataset_mu_and_sigma_test(df, n=None, columns=[]):
    """Take a columns labels and calculates the mean and standard deviation
    (std) for that given label checks if value x from label passes the
    threshold calledt, t = mean +/- n x std. The function will check if x>=t
    or x<=t. Function will always return mu and sigma test for n=2 and n=3.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The data frame that contains the values.

    n : int, float(object)
        The "n" parameters is the factor of times the std is multiplied with.
        The default setting is set to None

    columns : list(object)
        The "columns" is list object that accepts items that are string typed.
        The "columns" parameter is the label that is subjected to the test.
        If parameter left empty function call is triggered. The default value
        of "columns" is null or empty.

    Returns
    -------
    df2 : pandas.core.frame.DataFrame(object)
        The "df2" returns is the same "df" object with 2 or 3 new columns
        per given label in "columns".

    new_frame : pandas.core.frame.DataFrame(object)
        The calculated mean and standard deviation for the given columns
        labels in "columms".


    Raises
    ------
    ValueError
        If "columns" items is not found in df as a column label.

    See Also
    --------
    dataset_pick_columns : For more information about "columns" is equal to
                           null and function called.

    """
    # Local Global
    df2 = dataset_copy_frame(df)
    db_columns = df2.columns.tolist()
    n_factor=2
    sigmaxn = ('_{0}_x_sigmas').format(str(n))

    # Checking if "columns" is empty
    if not columns:
        columns, idx = dataset_pick_columns(df2, split='groupby')

    # Checking that columns are present in "df"
    columns_search = all([i in db_columns for i in columns])

    if not columns_search:
        db_set = set(db_columns)
        columns_set = set(columns)
        diff = columns_set.difference(db_set)
        text =("The labels given in 'columns' were not found in 'df'."
               "Labels found that differed between 'df' and columns "
               "was\n:{0}").format(str(diff))
        ValueError(text)

    # Creating frame with mean and standard deviation
    new_frame = pd.DataFrame(
        [df2[columns].mean(), df2[columns].std()],
        index=['Mean', 'std'])

    # While loop that calculates mean and sigmas
    while 4 > n_factor:
        for i in columns:
            df2[str(i)+'_'+str(n_factor)+'_x_sigmas'] = (
                df2[i].values >= new_frame[i]['Mean'] + (
                    n_factor * new_frame[i]['std'])) | (
                        df2[i].values <= new_frame[i]['Mean'] - (
                            n_factor * new_frame[i]['std']))
        n_factor = n_factor + 1

        if n_factor >4:
                break
    if n:
        for i in columns:
            df2[str(i)+sigmaxn] = (
                df2[i].values >= new_frame[i]['Mean'] + (
                    n * new_frame[i]['std'])) | (
                        df2[i].values <= new_frame[i]['Mean'] - (
                            n * new_frame[i]['std']))
    return df2, new_frame








def dataset_pairwise_distance_points(df_work, workcolumns):
    """ Takes data frame
    and calculates distances between points for data frame with length >= 5.

    Parameters
    ----------
    df_work : pandas.DataFrame (object)
        Input data frame structured as
        dataset_outliers return data frame.

    workcolumns : list
        The parameter 'workcolumns' the list of picked columns
        from dataset_outliers. An inhered value.

    Returns
    -------
    distance : list
        The parameter 'distance' contains a list
        of calculated distances between points.

    See Also
    --------
    dataset_outliers : For more information about 'df_work' and 'workcolumns'
    """

    # Coping data frame
    df = dataset_copy_frame(df_work)

    # List for appending points distances
    distance = []

    if len(df) < 5:
        length_x_y = np.random.random(len(df))
        [distance.append((i, 0, 0)) for i in length_x_y]
        return distance
    # Data frame length
    length = range(0, len(df), 2)

    for i in length:

        # Adding coordinates
        y = i + 1
        y2 = y + 1

        # Logic Circuit to avoid indexing error
        if y2 < length[-1]:
            y3 = y + 1
        if y < len(df):
            b = i + 1
        if y + 1 > len(df):
            y = len(df) - 1
            i = y - 1

        # Transpose data frame and slice by 3 objects
        dfT = df.iloc[[i, y]][np.arange(3)].T
        dfT2 = df.iloc[[b, y3]][np.arange(3)].T

        # Distance calculator
        pair = (np.sqrt(
            (dfT.loc[[workcolumns[0]]].values[0][1] -
                dfT.loc[[workcolumns[0]]].values[0][0]) ** 2 +
            (dfT.loc[[workcolumns[1]]].values[0][1] -
                dfT.loc[[workcolumns[1]]].values[0][0]) ** 2))
        between = (np.sqrt(
            (dfT2.loc[[workcolumns[0]]].values[0][1] -
                dfT2.loc[[workcolumns[0]]].values[0][0]) ** 2 +
            (dfT2.loc[[workcolumns[1]]].values[0][1] -
                dfT2.loc[[workcolumns[1]]].values[0][0]) ** 2))

        # Adding distance to list
        distance.append((pair, i, y))
        distance.append((between, b, y3))

    # Deleting the half point between outliers
    del distance[(len(df) / 2) - 1]

    # Modifying the list
    distance = distance[:-1]
    distance.insert(0, (0, 0, 0))
    distance.insert(len(distance), (0, 0, 0))

    # Returning list with distances
    return distance


def dataset_outliers_id_line_distance(df, num, columns=[], indexer=[]):
    """ Takes any given 2 columns from a pandas.core.frame.DataFrame(object)
    and calculates the distance to a theoretical identity line. Returns several
    data frames with a new column called 'distance_to_id_line'.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The given pandas object given by user.

    number : int(object)
        The parameter called 'number' is the number of points return from the
        top and bottom section e.g if the parameter is set to 5, the total
        amounts points return is 10 given there are 5 top and 5 bottom.

    columns, indexer : list(optional)
        The parameter is optional should however contain present column names
        found in 'df' parameter.

    Returns
    -------
    df_work1, df_subset : pandas.core.frame.DataFrame(object)
        Both returns are pandas.core.frame.DataFrame(object), however the
        return called 'df_work1' contains the distance for all given points
        from 'df' and the 'df_subset' the subset given by the parameter called
        'number' parameter.

    df_work2, df_zeros : pandas.core.frame.DataFrame(object)
        The parameter called 'df_work2' has all distance to identity line
        except from 0 values. The 'df_zeros' is all the point that has been
        calculated to 0 distances from identity line.

    work_columns, indexer : list(object)
        Both returns contains are list objects with string as items.

    See Also
    --------
    data_set.dataset_pick_columns : for more information about 'split'
                                    parameter.

    Raises
    ------
    AssertionError
        If lengths for 'work_columns' or 'indexer' differ 2 or 1.

    """
    # Locale globals
    orthogonal_to_id_line = np.array([1.0, -1.0])
    df_work1 = dataset_copy_frame(df)
    df_subset = pd.DataFrame()
    work_columns = columns
    indexer = indexer

    # Calculation of distance d from id line
    orthogonal_to_id_line /= np.sqrt((orthogonal_to_id_line ** 2).sum())

    # Check if columns and indexer is empty
    if not(work_columns and indexer):
        work_columns, indexer = dataset_pick_columns(df_work1, split='groupby')

    # Test for columns and indexer are equal to 2 and 1.

    try:
        assert len(work_columns) == 2
        assert len(indexer) == 1

    except AssertionError:
        print ("\n The columns picked to work with exceed the numbers"
               " of expected"
               " which is 2 or that indexer faces the same problem."
               " Expected columns are for is :\n 2 and"
               " user input was: {0}. "
               " Expected indexer is : \ 1 and"
               " user input was : {1}"
               "").format(len(work_columns), len(indexer))
        return

    # Calculates distance to identity line from given x and y values.
    df_work1['distance_to_id_line'] = orthogonal_to_id_line.dot(
            [df_work1[work_columns[0]].values,
                df_work1[work_columns[1]].values])

    # Sort data frame according to the ascending order in given column.
    df_work1 = df_work1.sort(columns='distance_to_id_line')

    # Removes Zero distance values
    df_work2 = df_work1[~(df_work1['distance_to_id_line'] == 0)]

    # Slice all distance equal to zero
    df_zeros = df_work1[df_work1['distance_to_id_line'] == 0]

    # Append top and bottom
    df_subset = df_subset.append(df_work2.head(num))
    df_subset = df_subset.append(df_work2.tail(num))
    df_subset = df_subset[[indexer[0], work_columns[0], work_columns[1]]]

    return df_work1, df_work2, df_subset, df_zeros, work_columns, indexer


def dataset_outliers_extrem_y_values(df, num, columns=[], indexer=[]):
    """ The function returns,  the y-value(s) extremes. The top
    'num' number(s) and  the  bottom 'num' numbers, with corresponding
    x-value(s) and all in ascending order.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input data frame.

    num : int (optional)
        The number of outliers returned is given
        by the 'num' parameter. Note that the number given
        is half of the number of outliers e.g. if num = 5
        then 5 bottom values and 5 top values are returned
        from input data frame.

    columns, indexer : list(optional)
        The parameter is optional should however contain present
        column names found in 'df' parameter.

    Returns
    -------
    df_work5 : pandas.DataFrame (object)
        The 'df_work5' is subset of values from input data frame.

    workcolumns : list
        The 'workcolumns' list of picked columns to extract top and
        bottom values.

    grouper : list
        The indexer for input data frame.

    See Also
    --------
    dataset_pick_columns : for more information about 'split' parameter.
    """
    # Global
    split = 'groupby'
    workcolumns = columns
    grouper = indexer
    # Copies data frame.
    df_work1 = dataset_copy_frame(df)

    # Check for columns or grouper is empty.
    if not(workcolumns and grouper):
        # Calls pick columns.
        workcolumns, grouper = dataset_pick_columns(df_work1, split=split)

    # Picks indexer and resets data frame.
    df_work2 = df_work1.set_index([grouper[i] for i in range(len(grouper))])
    df_work3 = df_work2[[workcolumns[i] for i in range(len(workcolumns))]]
    df_work3 = df_work3.reset_index()

    # Subset of values with top and bottom outliers.
    top = df_work3.iloc[df_work3[workcolumns[-1]].argsort()][-1 * num:]
    bottom = df_work3.iloc[df_work3[workcolumns[-1]].argsort()][:num]

    # Data frames added together.
    df_work4 = top.append(bottom)

    # Sort by ascending number for picked x column
    df_work5 = df_work4.sort(workcolumns[0])

    # Returns new data frame and columns names plus indexer.
    return df_work5, workcolumns, grouper


def dataset_outliers_midpoint(df, percentage, num, columns=[], indexer=[]):
    """ Takes absolute pairwise distances between 2 columns and compares to the
    mid point between x and y + threshold.
    ::

        Adds 4 new column names
        - 1. 'Abs_diff_x_and_y' = This is the absolute difference between x and
             y point.

        - 2. 'Threshold' = The mid point between x and y with the addition of
             percentage multiplied by the mid point. However the values are than
             used for the 'Mid_point_outlier' as calculations for sample mean
             estimates and standard deviation estimates for the 'population'.
             These parameters and the used like this  'Threshold_calc' is equal
             to ('Threshold sample mean' + 'Threshold standard deviation mean'
             x 2) calculations based of the 3 sigma rule [1] .

        - 3. 'Ratio_abs_and_threshold' = Which are the values in
             'Abs_diff_x_and_y'/'Threshold_calc'

        - 4.  'Mid_point_outlier' = It's column with boolean replies to if it
              passes an outlier or not. This is done by comparing values from
              columns in 'Abs_diff_x_and_y' >= 'Threshold_calc'

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The given pandas data frame.

    percentage : float(optional)
        The 'percentage' is a float that ranges from 0.0 to 1.0
        value. The input is always calculated as 1 - percentage
        e.g that input of 0.05 returns a 'percentage' = 0.95.

    number :  int
        The parameter called 'number' is the number of points
        return from the top and bottom section e.g if the parameter
        is set to 5, the total amounts points return is 10
        given there are 5 top and 5 bottom.

    columns, indexer : list(optional)
        The parameter is optional should however contain present
        column names found in 'df' parameter.

    Returns
    -------

    df_work, df_true : pandas.core.frame.DataFrame(object)
        The parameter called 'df_work' all data calculated from 'df'. The
        'df_true' is all values for 'df' that is 'mid point outlier'.

    df_subset : pandas.core.frame.DataFrame(object)
        The parameter called 'df_subset' is the 'num' of top and 'num' of
        bottom indices from 'df_work2'.

    df_work2, df_zeros : pandas.core.frame.DataFrame(object)
        The parameter called 'df_work2' has all absolute distance except 0
        values. The 'df_zeros' is all absolute distance that equal to 0.

    columnames, index : list(object)
        The contain string names for columns picked in 'df'.

    See Also
    --------
    data_set.dataset_pick_columns : For more information about 'split'
                                    parameter.

    References
    ----------
    .. [1] Wikipedia contributors. "68–95–99.7 rule." Wikipedia, The Free
           Encyclopedia. Wikipedia, The Free Encyclopedia, 26 Jun. 2016. Web. 6
           Jul. 2016.
    """
    # Local Global
    percentage = float(percentage)
    df_subset = pd.DataFrame()
    df_work = dataset_copy_frame(df)
    columnames = columns
    index = indexer

    # Check for empty columns and index.
    if not(columnames and index):
        columnames, index = dataset_pick_columns(df_work, split='groupby')

    # Check that percentage is between 0-1
    if not (percentage <= 1 and 0 <= percentage):
        prompt = (" The percentage should be float value"
                  " between 0-1 user entered :\n {0}\n\n").format(percentage)
        raise ValueError(prompt)
        return

    # Calculating percentage.
    percentage = float((1 - percentage) + 1)

    # Calculating the absolute difference between x and y coordinate.
    df_work['Abs_diff_x_and_y'] = np.sqrt(np.abs(
        (df_work[columnames[0]].values - df_work[columnames[1]].values) ** 2))

    # Calculating the mid-point of between the x and y + threshold.
    df_work['Threshold'] = (
        ((np.sqrt((
            np.abs(df_work[columnames[0]].values) +
            np.abs(df_work[columnames[1]].values)) ** 2)) / 2.0) * percentage)

    # Check that x and y difference is greater than threshold.
    df_work['Mid_point_outlier'] = (df_work['Abs_diff_x_and_y'].values
                                    > df_work['Threshold'].mean() + df_work[
                                        'Threshold'].std() * 2)
    # Adding Ratio column
    df_work['Ratio_abs_and_threshold'] = df_work[
        'Abs_diff_x_and_y'].values / (df_work['Threshold'].mean() +
                                      df_work['Threshold'].std() * 2)

    # Sort data frame according to the ascending order in given column.
    df_work = df_work.sort(columns='Abs_diff_x_and_y')

    # Slicing Mid point outlier values from main data frame.
    df_true = df_work[df_work['Mid_point_outlier'] == True]

    # Removes Zero distance values
    df_work2 = df_true[~(df_true['Abs_diff_x_and_y'] == 0)]

    # Slice all distance equal to zero
    df_zeros = df_work[df_work['Mid_point_outlier'] == False]

    # Append top and bottom
    df_subset = df_subset.append(df_work2.head(num))
    df_subset = df_subset.append(df_work2.tail(num))
    df_subset = df_subset[[index[0], columnames[0], columnames[1]]]
    return df_work, df_true, df_work2, df_zeros, df_subset, columnames, index


def dataset_value_transformer(df, split='groupby'):
    """Picked column(s) with value(s) are used as exponents
    for a base 2 exponentiation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame object from pandas.

    Returns
    -------
    df1_work1 : pandas.DataFrame
        Pandas data frame with transformed values.

    See Also
    --------
    dataset_pick_columns : For more information about 'split' parameter.
    """

    # Copying Data frame
    df_work1 = dataset_copy_frame(df)

    # Columns are picked
    columns_work, column_names = dataset_pick_columns(df_work1, split)

    # Columns to work with
    df2 = df_work1[[columns_work[i] for i in range(len(columns_work))]]

    # Transform values from columns
    df3 = np.power(2, df2)

    # Putting together a new values with old data frame
    for i in range(len(column_names)):
        df_work1[column_names[i]] = df3[column_names[i]].values

    # Returning new transformed data frame
    return df_work1


def dataset_add_sgd_goslimterms(SGD_GO_Slim_Terms, df, join_key='ORF'):
    """ Functions returns a new data frame with added GO terms.
    Data frames are joined as SQL ways inner. The joining is set
    from 1 to many.

    Parameters
    ----------
    SGD_GO_Slim_Terms : pandas.core.frame.DataFrame(object)
        The return value given by dataset_import_goterms

    df : pandas.core.frame.DataFrame(object)
        The 'df' input frame.

    join_key : str(optional)
        The 'join_key' parameter is key word the column name
        present in both 'SGD_GO_Slim_Terms' and 'df'.

    Returns
    -------
    df_merge : pandas.core.frame.DataFrame(object)
        The 'df_merge' is the new frame with GO slim terms.

    Raises
    ------
    ValueError
        If 'join_key' not present in columns.

    See Also
    --------
    dataset_import_goterms : For more information on how to get the parameter
                             called 'SGD_GO_Slim_Terms'.
    """
    # Copy Frame
    df_work1 = dataset_copy_frame(df)

    # Column names
    SGD_columns = SGD_GO_Slim_Terms.columns.tolist()
    df_work1_columns = df_work1.columns.tolist()

    try:
        SGD_columns.index(join_key)
        df_work1_columns.index(join_key)

    except ValueError:
        print ("\n Columns present in 'SGD_GO_Slim_Terms' are :\n"
               "\n\t {0}"
               "\n\n Columns present in 'df' are :\n"
               "\n\t {1}"
               "\n\n Joining key for merging frames was:\n"
               "\n\t {2}").format(SGD_GO_Slim_Terms,
                                  df_work1_columns, join_key)

    # Merge by key
    df_merge = pd.merge(df_work1, SGD_GO_Slim_Terms, on=join_key)

    # Returns merge frames
    return df_merge


def dataset_add_goslimterms(
        df, index_name="Strain", filepath="./yeastmine", delimiter="\t"):
    """Function adds data from a file called 'yeastmine.csv'. Outputs dictionary
    with gene ontology (GO) slim terms combined with a pandas data frame
    object. The csv-file columns are renamed as 'Standard_Names' for gene
    names, 'Systematic_Names' for the locus naming standard applied by SGD,
    'GO_Slim_Terms' as the GO slim term, 'Database' for the data base source of
    the GO-term and the GO unique code identifier was labelled as the
    'GO_identifier'. The csv-file containing the GO terms must be 'TAB'-
    delimited file and must be named 'yeastmine.csv'.  The file is imported
    by locating the 'yeastmine.csv' in same directory as the current working
    directory. The Function returns a data frame with GO-annotations and
    the imported 'yeastmine.csv' file.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The pandas data frame that is to be combined with GO-annotation
        located in yeastmine.csv

    index_name : str(optional)
        The parameter called 'index_name' specifies a column name present in
        'df' that contains the standard nomenclature for loci used by SGD e.g
        YAL002W (VPS8). The default value for 'index_name' is set to "Strain".

    filepath : str(optional)
        The 'filepath' is the relative path from current working directory
        to the csv-file containing the GO-annotations. The default value of
        'filepath' is set to './yeastmine.csv'.

    delimiter : str(optional)
        The parameter called 'delimiter' is how the csv-file-import read.
        The 'delimiter' parameter is set to read as 'TAB'-delimited.

    Returns
    -------
    df : pandas.core.frame.DataFrame(object)
        A new frame with GO annotations.

    table : pandas.core.frame.DataFrame(object)
        The csv-file containing the GO-annotations called 'yeastmine.csv'.

    See Also
    --------
    dataset_import_goterms : For alternative way of GO-annotating data in 'df'

    dataset_add_sgd_goslimterms : For alternative way of GO-annotating data
                                  in 'df'

    Notes
    -----
    The function was scrapped for the above alternative ways to annotate.

    """

    # Clear column axis of unwanted columns
    df = dataset_clear_columns(df)

    # Import of file
    column_names = ["Standard_Names", "Systematic_Names",
                    "GO_Slim_Term", "Database", "GO_identifier"]

    table = pd.read_csv(filepath, delimiter=delimiter,
                        header=False, names=column_names)

    # Copy table
    table1 = table.copy()

    # Creating an new Dataframe object
    goSlimterm = pd.DataFrame(table1)

    # Slice and turn into list types
    names = list(goSlimterm["Systematic_Names"])
    goTerms = list(goSlimterm["GO_Slim_Term"])
    database = list(goSlimterm["Database"])

    goslimterm = defaultdict(list)
    dataBase = defaultdict(list)

   # Appending to object given lists
    for i, item in enumerate(names):
        goslimterm[item].append(goTerms[i])
        dataBase[item].append(database[i])

    # Constructing Dictionaries
    goslimterm = {k: v for k, v in goslimterm.items() if len(v) > 1}
    dataBase = {k: v for k, v in dataBase.items() if len(v) > 1}

    # Container for Dictionaries
    result_goTerm = []
    result_database = []

    # Names for data frame of interest
    index1 = list(df[index_name])

    # Container for data not found message
    txt = ["Uncategorized"]

    # Creating new columns in Dataframe
    for i in range(len(df[index_name])):
        var = set(goslimterm.get(str(index1[i]), txt))
        var2 = set(dataBase.get(str(index1[i]), txt))
        result_goTerm.append(var)
        result_database.append(var2)
    df["GOslimTerms"] = result_goTerm
    df["Database"] = result_database
    result_goTerm[:] = []
    result_database[:] = []
    return df, table


def dataset_filtering(
        df, split='groupby', filename='filtered_data_changeme',
        filtertype='any'):
    """Harsh filtering of column(s) data by label(s).All label(s) containing
    NAN or inf values are discarded.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame object from pandas.

    filtertype : str, optional
        The filtertype option determine how NaN or inf values are processed
        if string value is 'any' discards labels from data frame for any
        presences of NAN or inf.
        If string value is 'all' labels a dropped if all values are NaN or inf
        (The default for 'filtertype' is any).

    Returns
    -------
    df2 , raw_data : pandas.DataFrame, float
        The function returns a new data frame and a float.

    See Also
    --------
        dataset_pick_columns : For more information about 'split' parameter.
        dataset_filesave : For more information about 'filename' parameter.
    """

    # Copying data frame
    df_work1 = dataset_copy_frame(df)
    # Gets picked columns back
    columns_work, grouper = dataset_pick_columns(df_work1, split)

    print "Columns to work with:" + "\n", columns_work

    # Check data points
    raw_data = float(len(df_work1[grouper[0]])) * 1.0

    # Filtering data frame on columns_work
    df2 = df_work1.replace(
        [np.inf, -np.inf], np.nan).dropna(
            subset=[columns_work[i] for i in range(
                len(columns_work))], how=filtertype)

    # Creates csv-file for backup
    dataset_filesave(df2, filename)

    # Prompt to look for specific file in directory
    return df2, raw_data


def dataset_stats_values(df, split='groupby', filename='stats_values', ddof=1):
    """Calculates mean, variance(var), standard deviation (std)  per label(s)
    from a data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame from pandas
    ddof : int or float
        Numbers sets the degree of freedom of statistical calculation.
        The 'ddof' is set to 1 by default assumes sample statistics

    Returns
    -------
    df3 : pandas.DataFrame
        New data frame with mean values for given groups.

    See Also
    --------
    dataset_pick_columns : For more information about 'split' parameter.
    dataset_filesave : For more information about 'filename' parameter.
    """

    # Copy given data frame
    df_work1 = dataset_copy_frame(df)

    # Get rid of unwanted column
    cols_picked, grouper = dataset_pick_columns(df_work1, split)

    # Making new dataframe
    df2 = df_work1[cols_picked]
    df3 = df2.groupby([grouper[i] for i in range(len(grouper))])
    df3 = df3.agg([np.mean, np.std, np.var, np.median], ddof=ddof)

    # Resets index
    df3 = df3.reset_index()

    # Renaming column names
    columns = [list(names) for names in df3.columns.values]
    columns_changed = [i[0] + '_' + i[1] for i in columns]
    columns_changed = [i.rstrip('_') for i in columns_changed]
    df3.columns = columns_changed

    # Return save file to working directory
    dataset_filesave(df3, filename)

    # Returns data frame!
    return df3


def dataset_pvalue(
        df, split='groupby', filename='p_values_per_group',
        new_column='P-value', welch_test=False):
    """Calculates p-values for two conditions group wise. The p-value is
    calculated by scipy.stats.ttest_ind function. The function assume not equal
    variance and sample size.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame from function returned called
        dataset_merge_by_condition.

    new_column : str, optional
        Column name for obtain p-values.
        The default values for 'new_column' is set to 'P-value').

    welch_test : boolean(optional)
        The 'welch_test' parameter is set to 'False' assuming
        unequal sample sizes and variance per test.

    Returns
    --------
    df2, dict_df2, Grouper : pandas.DataFrame, dict, str
        New 1 x 2 data frame with label column and p-value column.
        Also 'dict_df2' contains a dictionary with the grouper as key
        and p-values as items.The 'Grouper' is str.

    See Also
    --------
    dataset_pick_columns : For more information about 'split' parameter.

    dataset_test_for_Sample_Variance_Between_EXP_and_Control : For more
                                                               information
                                                               about check for
                                                               variances.

    dataset_filesave : For more information about 'filename' parameter.

    """

    # Data frame copied
    df_work2 = dataset_copy_frame(df)

    # Calls function and retruns two variables.
    columns, grouper = dataset_pick_columns(df_work2, split)

    # Creates dictionary from columns
    dict_columns = {i: columns[i] for i in range(len(columns))}

    # Prompts user of columns picked
    pprint(dict_columns)

    # Accepts user inputs and prompts information.
    print ("\n Pick the column with boolean values, followed by"
           "\n the column containing data for the t-test."
           "\n\n Example \n\t1:Boolean Values , 10:Values"
           "\n\t Enter: 1,10,"
           "\n Press Enter key")
    pick_columns = list(input('Column order picked:\t'))

    # Calling check for zero variance function
    df_work1 = dataset_test_for_Sample_Variance_Between_EXP_and_Control(
        df_work2, dict_columns=dict_columns,
        pick_columns=pick_columns, grouper=grouper)

    # Prompts user choice
    print dict_columns.get(pick_columns[0]), dict_columns.get(pick_columns[1])

    # Creates data frames with pick columns.
    df2 = pd.DataFrame(
        [(g[grouper[0]].values[0],
            ttest_ind(
                g[g[dict_columns.get(
                    pick_columns[0])]][dict_columns.get(pick_columns[1])],
                g[g[dict_columns.get(
                    pick_columns[0])] == False][dict_columns.get(
                        pick_columns[1])], equal_var=welch_test)[1])
            for _, g in df_work1.groupby(grouper[0])],
        columns=[grouper[0], new_column])

    # Save a file copy of new data frame
    dataset_filesave(df2, filename)

    # Creates a dictionary
    dict_df2 = dataset_to_dict(df2, keys=grouper[0], values=new_column)

    # Variable
    Grouper = grouper[0]

    # Returns data frame, dict of p-value and grouper.
    return df2, dict_df2, Grouper


def dataset_test_for_Sample_Variance_Between_EXP_and_Control(
        df, dict_columns={}, pick_columns=[], grouper=[]):
    """ Function takes the sample variance from 2 groups
    the Experimental and the Control. If sample variance for both groups
    is 0 it is excluded from frame.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The given data frame from user.

    dict_columns : dict(object)
        A set of int keys where values are column names present in 'df'.
        dict_columns = {1: Column_Name}

    pick_columns : list(object)
        List object where items corresponds to integer keys values
        in 'dict_columns'.

    grouper : list(object)
        List object where items is column name present in 'df'.
        Used as indexer for grouping values together e.g
        systematic orf names or other flag.

    Returns
    -------
    df_work2 : pandas.core.frame.DataFrame(object)
        Pandas data frame excluded for variances equal to 0.

    Saves 2 <TAB> delimited csv files in working directory named
        Sample_Variance_in_Experiment.csv
        Sample_Variance_in_Control.csv
    """
    # Copying input frame
    df_work1 = dataset_copy_frame(df)

    # Getting Names for Sample Variance
    orf_names = [
        [g[g[dict_columns.get(pick_columns[0])]][grouper[0]].unique()]
        for _, g in df_work1.groupby(grouper[0])]

    df_work1_var = pd.DataFrame(
        [[g[g[dict_columns.get(
            pick_columns[0])]][dict_columns.get(pick_columns[1])].var(),
            g[g[dict_columns.get(
                pick_columns[0])] == False][dict_columns.get(pick_columns[1])].var()]
            for _, g in df_work1.groupby(grouper[0])],
        columns=['Sample_Variance_in_Experiment',
                 'Sample_Variance_in_Control'])

    # Adding Sample names
    df_work1_var[grouper[0]] = [str(i[0]) for i in orf_names]

    # Sub sampling on conditions where variance is zero.
    df_work1_var_zeros = df_work1_var[
        (df_work1_var.Sample_Variance_in_Experiment == 0) &
        (df_work1_var.Sample_Variance_in_Control == 0)]

    # Boolean vector use for exclusion from main data frame.
    exclusion_array = np.ravel(df_work1_var_zeros[grouper[0]].values)

    boolean_vector = df_work1[grouper[0]].isin(exclusion_array)
    inverted_boolean = np.invert(boolean_vector)

    # Getting data frame with variance
    df_work2 = df_work1[~boolean_vector]

    # Getting data frame with zero variance
    df_exclueded = df_work1[~inverted_boolean]

    # Exporting frames as Tab delimited csv-files
    dataset_filesave(
        df_work2, filename='Table_with_Variance_between_EXP_and_Control')
    dataset_filesave(
        df_exclueded, filename='Zero_Variance_between_EXP_and_Control')
    return df_work2


def dataset_combine_2_frames_to_1(dataframe_list=[], axis=1):
    """
    The function concat 2 pandas data frame into 1.

    Parameters
    ----------
    dataframe_list : list(object)
        The parameter called 'dataframe_list' accepts
        only panda.core.frame.DataFrame(object) as
        valid entries.

    axis : int(optional)
        The parameter called 'axis' is optional
        and has length of 1 and accept either 0 or 1
        as valid inputs. The parameter is set to
        1 as default which is creating the new data
        by adding indices columns wise.

    Returns
    -------
    df_without_NANs : pandas.core.frame.DataFrame(object)
        New combine object from 'dataframe_list'.

    Raises
    ------
    TypeError
        If items in dataframe_list is not a
        pandas.core.frame.DataFrame(object)

    ValueError
        If the 'axis' parameter differ from integer type
        or has a value other than 0 or 1.
    """
    # Locale Global
    frame_type = type(pd.DataFrame())
    types_found = [type(i) for i in dataframe_list]
    axis_type = type(int())

    # Check that indices are pandas data frame objects
    if not (len(dataframe_list) == sum([
            type(i) == type(pd.DataFrame())
            for i in dataframe_list])):

        # Prompt user information
        prompt = ("1 or more items in 'dataframe_list' is not"
                  " a pandas data frame object"
                  " typed as :\n\n{0}\n\n"
                  " and found types"
                  " were \n\n{1}\n\n").format(frame_type, types_found)
        raise TypeError(prompt)
        return

    # Check that axis is right type and value
    if not(axis == 0 or axis == 1) or not(type(axis) == axis_type):
        prompt2 = (" Parameter 'axis' only accepts values that are"
                   " have the type {0} and values 0 or 1"
                   " user entered : {1}").format(axis_type, axis)
        raise ValueError(prompt2)
        return

    # Assembling the 2 frames to 1
    df = pd.concat(dataframe_list, axis=axis)

    # Getting filling NaNs with value
    df_without_NANs = dataset_fill_NAN_with_any_value(df, mask='Empty')

    # Returning new data frame.
    return df_without_NANs


def dataset_merge_by_condition(
        df, dataframe_list=[], split1='indexby', split2='groupby',
        state1='NaCl', state2='Basal', phenotype='',
        condition='State3', condition2='State2',
        condition_true='NaCl', new_column='State', filename='merge_dataframe'):
    """Data frame is split up by features for e.g salt milieu, non salt milieu
    and phenotype. First round of choice splits data frame in two, second round
    of choice puts it together again.

    Parameters
    ----------
    df : pandas.Dataframe
        Data frame input to function

    dataframe_list : list(object)
        The parameter called 'dataframe_list' accepts
        only panda.core.frame.DataFrame(object) as
        valid entries.

    state1, state2 : str, optional.
        The parameters 'state1' and 'state2' are to given conditions.
        (The default values for 'state1' is 'NaCl' and for
        'state2' is 'Basal').

    phenotype : str
        String labelling what character is worked on.

    condition, condition2 : str, optional.
        Column names for 'state1' and 'state2' values.
        (The default values for 'condition' is 'State3'
        and condition2 is 'State2').

    condition_true : str, optional
        The name for the boolean column that check condition.
        (The default value for 'condition_true' is 'NaCl').

    new_column : str, optional
        Labels for columns with mixed conditions.
        (The deafault value for 'new_column' is 'State').

    Returns
    -------
    df_work4 : pandas.Dataframe
        Data frame with new layout and two new columns

        Also saves a back up file produced Dataframe.

    See Also
    --------
        dataset_pick_columns : For more information
        about parameters 'split1' and 'split2'.

        dataset_filesave : For more information about 'filename' parameter.

    Notes
    ------
    This function can work with max two columns per run and one phenotype!

    """

    print "\n The function takes at most 2 condition e.g Salt and No Salt!",
    print "\n Users can only work with one phenotype per run.\n",

    print ("\n Data frame split in 2 by user:"
           "\n 1# Round pick indexer(s) and 1 Column for condition e.g. Salt"
           "\n Press enter key"
           "\n Redo step 1# except that condition is Non-salt or other\n\n")

    # Call function if needed
    if dataframe_list:
        df_work1 = dataset_combine_2_frames_to_1(dataframe_list=dataframe_list)

    # Creates a copy of input data frame
    else:
        df_work1 = dataset_copy_frame(df)

    # Dividing data frame by two.
    df_work2, indexname2 = dataset_pick_columns(df_work1, split=split1)
    df_work3, indexname3 = dataset_pick_columns(df_work1, split=split1)

    # Resetting index
    df_work2 = df_work2.reset_index()
    df_work3 = df_work3.reset_index()

    # Checking how many columns
    len_work2 = len(df_work2.columns.tolist())
    len_work3 = len(df_work3.columns.tolist())

    # Creating new columns
    df_work2.insert(
        len_work2, condition + '_' + state1,
        state1, allow_duplicates=True)
    df_work3.insert(
        len_work3, condition2 + '_' + state2,
        state2, allow_duplicates=True)

    # Adding data frames together
    df_work4 = df_work2.append(df_work3)

    print ("\n User has 2 input rounds example of how:"
           "\n 1# - Pick an empty column for Salt and Non-Salt column"
           "\n\t Press enter\n"
           "\n 2# - Chose values for a certain phenotype with corresponding"
           "\n features e.g. Lag values in Salt, followed by"
           "\n followed by Lag values in Non-salt!"
           "\n\t Press enter\n")

    grouper1, grouper2 = dataset_pick_columns(df_work4, split=split2)

    # Putting things together
    df_work4[new_column] = pd.concat([df_work4[grouper1[0]].dropna(),
                                      df_work4[grouper1[1]].dropna()])

    df_work4[phenotype] = pd.concat([df_work4[grouper2[0]].dropna(),
                                     df_work4[grouper2[1]].dropna()])

    # Deleting other old columns
    del df_work4[grouper1[0]]
    del df_work4[grouper1[1]]
    del df_work4[grouper2[0]]
    del df_work4[grouper2[1]]

    # Checking Lengths for new data frame
    len_work4 = len(df_work4.columns.tolist())

    # Insert Boolean column thats checks condition
    df_work4.insert(
        len_work4, condition_true,
        df_work4[new_column].values == state1,
        allow_duplicates=True)

    if dataframe_list:

        # Localize column names with NAN values.
        columns_with_NANs = [
            i for i in df_work4.columns.tolist() if sum(df_work4[i].isnull())]

        # Merge columns to 1 index and remove NANs
        df_work4['Merge_index'] = pd.concat(
                [df_work4[i].dropna() for i in columns_with_NANs])

        # Drop columns names that had NANs in them.
        df_work4 = df_work4.drop(
            labels=[i for i in columns_with_NANs], axis=1)

        # Slice of NAN with flag empty
        df_work4 = df_work4[df_work4['Merge_index'] != 'Empty']

        print 'hello'
    # Save as file
    filename = filename + phenotype
    dataset_filesave(df_work4, filename)

    # Returns data frame.
    return df_work4


def dataset_top_and_bottom_extremes(
        df, num, percentage, columns=[], indexer=[], func_call=[]):
    """ Function calls different outliers function and assembles
    the result in to new pandas data frame.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input data frame.

    num : int (optional)
        The number of outliers returned is given by the 'num' parameter. Note
        that the number given is half of the number of outliers e.g. if num = 5
        then 5 bottom values and 5 top values are returned from input data
        frame.

    percentage : float(optional)
        The 'percentage' is a float that ranges from 0.0 to 1.0 value. The
        input is always calculated as 1 - percentage e.g that input of 0.05
        returns a 'percentage' = 0.95.

    columns, indexer : list(optional)
        The parameter is optional should however contain present column names
        found in 'df' parameter.

    func_call : list(optional)
        The parameter call func_call is list with 3 items.
        The parameter accepts only True or False statements.

    Returns
    -------
    df_results : pandas.core.frame.DataFrame(object)
        The 'df_results' is sort in ascending matter from values found in
        'columns' first item.

    columns, indexer : list()
        The contains given column names for outlier(s) calculations.

    func_name : list()
        The list object containing the name for function called.


    See Also
    --------
    dataset_outliers_extrem_y_values : for more information about 'subset'

    dataset_outliers_id_line_distance : for more information about 'subset2'

    dataset_outliers_midpoint : for more information about 'subset3'

    Notes
    ------
    About function is that created frame is additive per function of outlier(s)
    found. As well as the order of indices in 'columns' should be '[X-values,
    Y-values]'.

    """

    print (" \nFriendly reminder for 'columns' inputs is:"
           " \n'columns' = ['X-values', Y-values]\n")
    # Global Local
    func_name = []
    subset = []
    subset2 = []
    subset3 = []
    call_dict = {
        'Extreme Y-values outliers': False,
        'Identity Line distance outliers': False,
        'Midpoint outliers': False}

    # Copy frame
    df_work1 = dataset_copy_frame(df)

    # Empty columns and indexer
    if not(columns and indexer):
        columns, indexer = dataset_pick_columns(df_work1, split='groupby')

    # Empty function call
    if not func_call:
        func_call = [False, False, False]

    # Updating call_dict
    for i in range(len(call_dict)):
        keys = call_dict.keys()
        keys.sort()

        # Updated
        call_dict[keys[i]] = func_call[i]

    # Function call

    if call_dict.values()[0]:
        print "\nUser has implemented this:{0}".format(call_dict.keys()[0])
        subset, col1, idx1 = dataset_outliers_extrem_y_values(
            df_work1, num, columns=columns, indexer=indexer)

        # Unwanted returns removed
        del col1, idx1

        # Appending func_call to func_name
        func_name.append(call_dict.keys()[0])

    if call_dict.values()[1]:
        print "\nUser has implemented this:{0}".format(call_dict.keys()[1])
        A, B, subset2, zeros, col2, idx2 = dataset_outliers_id_line_distance(
            df_work1, num, columns=columns, indexer=indexer)

        # Unwanted returns removed
        del A, B, zeros, col2, idx2

        # Appending func_call to func_name
        func_name.append(call_dict.keys()[1])

    if call_dict.values()[2]:
        print "\nUser has implemented this:{0}".format(call_dict.keys()[2])
        C, D, E, zeros2, subset3, col3, idx3 = dataset_outliers_midpoint(
            df_work1, percentage, num, columns=columns, indexer=indexer)

        # Unwanted returns removed
        del C, D, E, zeros2, col3, idx3

        # Appending func_call to func_name
        func_name.append(call_dict.keys()[2])

    df_subset = [subset, subset2, subset3]
    df_subset = [i for i in df_subset if len(i)]

    if not df_subset:
        print "No subsets were created from 'df'. "
        return

    # Assembling results
    df_results = pd.concat(df_subset, axis=0)

    # Dropping duplicates.
    df_results = df_results.drop_duplicates()

    # Sorting by values in first given column.
    df_results = df_results.sort(columns=columns[0])

    return df_results, indexer, columns, func_name


def dataset_calculate_p_value_from_stats(
        merged_df, welch_test=False, sort_column=[], save_csv=False):
    """The function calculates p-values from the descriptive statistics such as
    mean, std and number observation made. The p-values are calculated by
    comparing the descriptive statistics and assuming that means are
    independent from each other. The t-test is a two-sided and the null
    hypothesis is that samples are identical. The function assumes that
    variance between samples is not equal.

    Parameters
    ----------
    merged_df : pandas.core.frame.DataFrame(object)
        The 'merged_df' parameter is a joined pandas data frame. The
        'merged_df' as two separate sources and formed as the 'df_merged'
        return from phen_o_links.dataassembler.data_assembler_merge_frames.

    welch_test : boolean(optional)
        The 'welch_test' specifies if p-values calculated are of equal
        variance between samples. The default value for 'welch_test' is False,
        which assumes equal variance between samples.

    sort_column : list(optional)
        The 'sort_column' specifies a column label in 'merged_df'. The
        'sort_column' accepts string as a valid input. The minimum requirements
        for 'sort_column' is a index value for the calculated p-values found
        in 'merged_df' and column label to sort the 'merged_df'. If left
        empty function call is triggered.

    save_csv : boolean(optional)
        The 'save_csv' executes to save return values as csv-files at
        current working directory. Delimiter for csv-file is set to 'TAB'.

    Returns
    -------
    merged_df2 : pandas.core.frame.DataFrame(object)
        The return value 'merged_df2' is the entered input 'merged_df' with
        the additional columns called 'P_index' and 'P_values'.

    csv : files(optional)
        The function may save the result from 'merged_df2' as a csv-file at
        the current working directory if 'save_csv' is set to True. Files
        created are labelled as 'stats_data_with_p_values.csv' which is the
        'merged_df2' return and the 'table_with_only_p_values.csv' file that
        only contain the calculated p-values and t-values.

    Raises
    ------
    ValueError
        If 'sort_column' input is not a string and If input given in
        'sort_column' is not present in 'merged_df' as a column label.

    See Also
    --------
    dataset_pick_columns : For more information about function call made when
                           'sort_column' is left empty

    dataset_filesave : For more information about the function that 'save_as'
                       call to create csv-files.

    phen_o_links.dataassembler.data_assembler_merge_frames : For more information
                                                           about how
                                                           'merged_df' input
                                                           is created.

    scipy.stats.ttest_ind_from_stats : For more information about 'welch_test'
                                       parameter and how t-test and p-values
                                       are performed and calculated.

    """
    # Local global
    sep_as = 'groupby'
    p_vals = []
    welch_test = not welch_test
    df_columns = merged_df.columns.tolist()
    if not sort_column:
        text = ("\nPick a minimum of 2 columns to sort data in 'merged_df'!\n "
                "Press Enter!"
                "Pick 1 column labels to be indexer for p-values "
                "Press Enter!")
        print text
        column, index = dataset_pick_columns(merged_df, split=sep_as)

    if sort_column:
        # Check for that entries are string.
        try:
            if not all(isinstance(i, str) for i in sort_column):
                raise ValueError
        except ValueError:
            text = ("The input in 'sort_column' was not only strings!")
            print text
            return
        try:
            [df_columns.index(i) for i in sort_column]

        except ValueError:
            text = ("The strings entered in 'sort_column' are not found in "
                    "'merged_df' as column labels. User entered {0} and "
                    "columns found in {1}.").format(sort_column, df_columns)
            print text
            return

        column = sort_column[1:]
        index = sort_column[:1]

    merged_df2 = merged_df.sort_values(by=column)

    text = ("P-values Calculation from stats!")
    print text

    text2 = ("\nThe first round pick the columns that corresponds for the "
             "condition 1. Pick the columns in following order:\n "
             "\n 'Sample mean', 'Standard deviation' and 'the number of "
             "observations'.")
    print text2

    work, index2 = dataset_pick_columns(merged_df2, split=sep_as)

    text3 = ("\nThe first round pick the columns that corresponds for the "
             "condition 2. Pick the columns in following order:\n "
             "\n 'Sample mean', 'Standard deviation' and 'the number of "
             "observations'.")
    print text3
    work2, index3 = dataset_pick_columns(merged_df2, split=sep_as)

    del index2, index3

    for i in range(len(merged_df2)):
        t, p = ttest_ind_from_stats(
            merged_df2[work[0]].values[i], merged_df2[work[1]].values[i],
            merged_df2[work[2]].values[i], merged_df2[work2[0]].values[i],
            merged_df2[work2[1]].values[i], merged_df2[work2[2]].values[i],
            equal_var=welch_test)
        idx = merged_df2[index[0]].values[i]
        pl = merged_df2[column[0]].values[i]
        p_vals.append((pl, idx, t, p))

    # Creating Frame
    pvalues = pd.DataFrame(
        data=p_vals, columns=[
            'P_' + column[0], 'P_' + index[0], 'T_values', 'P_values'])

    merged_df2["P_index"] = pvalues['P_' + index[0]].values
    merged_df2["P_values"] = pvalues["P_values"].values

    if save_csv:
        dataset_filesave(merged_df2, filename='stats_data_with_p_values')
        dataset_filesave(pvalues, filename='table_with_only_p_values')

    return merged_df2


if __name__ == "__main__":
    # Execute only as script
    print "Please import module named {0} with Ipython".format(__name__)
