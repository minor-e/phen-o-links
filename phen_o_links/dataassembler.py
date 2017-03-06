#!/usr/bin/env python

# This in alpha state no promises or refunds for data lost
# This script only work if all csv file are in the same
# directory.All filenames must be in descending order!!

import glob
import os
import re
import pandas as pd
import numpy as np
import dataset as ds

# Archaic functions
def data_assembler_csv():
    """This is a data frame assembler. Takes multiples tab delimited csv files
    and builds one csv file containing all data within a directory. This
    function is archaic and only functions with csv-files.

    Returns
    -------
    csv file : file
        A file called 'assembled_changeme.csv' in current working directory.

    """

    # Marks to way home
    path = r'./'

    # Takes all files with csv suffix

    allFiles = glob.glob(path + '*.csv')

    # Creates container for dataframe and data

    data_frame = pd.DataFrame()
    list_for_data = []

    # Creates iterator for list of file names

    for files in allFiles:

        # Reads csvs files

        dataframe = pd.read_csv(files, delimiter="\t")

        # Appends data to list from data frames

        list_for_data.append(dataframe)

    # Creates New Data Frame via concat command

    data_frame = pd.concat(list_for_data)

    # Creating column list
    txt = dataframe.columns.tolist()

    print txt

    # Export new csv file with data sets

    data_frame.to_csv('./assembled_changeme.csv', sep='\t', cols=txt)

    # Shows result and reminds user to change name of file

    print data_frame
    print "\nDon't forget to change the name of assembled_changeme.csv"

    # Deleting all elements

    del list_for_data[:]

    # Gets rid of elements in list!

    if len(list_for_data) > 0:
        list_for_data = []


# Mores advance functions
def data_assembler_II_multiple_files_import_CSV_format(
        sep='\t', suffix='.csv', flag=False, clear_by=None):
    """Constructs a file containing multiple csv files. Users should gather all
    wanted files in one  directory and navigate to that directory. Files should
    be numerated in ascending order to work correctly. However if user has
    labelled files as following with 'words' +'_'+ 'plate order'+'_'.
    The function will be able to sort the files correctly.

    Parameters
    ----------
    sep : str(optional)
        The parameter called 'sep' determines the delimiter in use for the csv
        file assembly. The parameter is set to be 'TAB'-delimited by default.

    suffix : str(optional)
            The parameter called 'suffix' is the given input for file
            specification i.e. file-'.txt' or other.

    flag : boolean(optional)
         The 'flag' operator is only viable if the file structure follows the
         pattern 'word'+'_'+'plate order'+'_'.

    clear_by : str(optional)
        The parameter called 'clear_by' takes one column header and removes
        extra rows due to nan or extra space.

    Returns
    -------
    new_frame : pandas.core.frame.DataFrame(object)
        The newly assembled csv.

    assemble_order : pandas.core.frame.DataFrame(object)
        The order of how the csv's other text files was concatenated. From 0
        to the last file.

    Raises
    ------
    AssertionError
        If first 'csv' file in list lacks the suffix of csv.

    ValueError
        If 'clear_by' value is not found in columns labels.

    Notes
    -----
        The flag option is only viable if dates in file names are put after
        plate order. Since the regular expression given can't distinguish
        pattern date pattern in file name.
    """

    # Locale global
    new_data_frame = pd.DataFrame()
    files_suffix = []
    new_order = []
    index_nr = []
    assemble_order=dict()

    # File selection
    path = os.getcwd()
    files_list = os.listdir(path)
    files = [i for i in files_list if os.path.isfile(str(i))]

    # Check for suffix and if files contain digits
    files_suffix = [i for i in files if i.endswith(suffix)]
    files_suffix_with_nr = [i for i in files_suffix if re.findall("\d", i)]

    # If flag True
    if flag:
        tmp = []
        for i in files_suffix_with_nr:
            tmp.append(re.findall("(\B[^a-z]\S?\d\S\B)", i))

        # Getting 1st item in tmp
        order_files = [int(tmp[i][0].replace("_","")) for i in range(len(tmp))]

        # zipping objects
        new_order = zip(order_files, files_suffix_with_nr)
        new_order.sort()

        # Creating data frame from zipped objects
        new_files = pd.DataFrame(new_order)

        # Files in ascending order
        files_suffix = new_files[1].tolist()

    # Check that for length
    if not (len(files_suffix) == len(files_suffix_with_nr)):
        index_nr = raw_input('Do you wish to '
                             'index \n files with digits?: press Y/N:\t')
        index_nr = str(index_nr)
    elif flag:
        index_nr = 'Y'
    else:
        files_suffix = files_suffix_with_nr
        index_nr = 'N'

    # Adding flag to files present in path
    if not files_suffix_with_nr or index_nr.lower() == 'y':
        n = 0
        if flag:
            n = 1
        new_names = ["%02d" % (i + n) + '_' + str(files_suffix[i])
                     for i in range(len(files_suffix))]
        [os.rename(files_suffix[i], new_names[i])
         for i in range(len(new_names))]
        files_suffix = new_names
    files_suffix.sort()

    # Concating files and adding plate order
    for filenames in range(len(files_suffix)):
        print "\n File concatenated as {0} : {1} \n".format(
            (filenames + 1), files_suffix[filenames])
        print filenames
        assemble_order.update({filenames:files_suffix[filenames]})
        plate_nr = filenames + 1
        temp_file = pd.read_csv(files_suffix[filenames], delimiter=sep)
        temp_file['Plate_Nr'] = plate_nr
        new_data_frame = new_data_frame.append(temp_file)

    # Function call to clear empty columns
    new_frame = ds.dataset_clear_columns(new_data_frame)

    # Clearing extra rows
    if clear_by:
        rm_empty = 1
        columns = new_frame.columns.tolist()
        try:
            columns.index(clear_by)

        except ValueError:
            text = "Empty cells are not removed"
            print text
            rm_empty = 0

        # Clearing extra rows!
        if rm_empty == 1:
            not_empty = ~new_frame[clear_by].isnull()
            new_frame["NotEmpty"] = not_empty
            new_frame = new_frame[new_frame["NotEmpty"] == True]
            del new_frame["NotEmpty"]

    # Creating pandas series with the file(s) order
    csv_order = pd.Series(assemble_order, index=assemble_order.keys())
    csv_order = csv_order.reset_index()
    csv_order.columns = ["Input_order_of_csv","Filename"]

    # Export new csv file with data sets
    new_frame.to_csv('./newfile_changeme.csv', sep=sep, index=False,
                     na_rep=np.nan)
    csv_order.to_csv('assemble_order_of_csv_files.csv',sep=sep, index=False)

    # Shows result and reminds user to change name of file
    print ("\nDon't forget to change the name of newfile_changeme.csv "
           "and assemble_order_of_csv_files")

    return new_frame


def data_assembler_creating_subset(df, column_name, pattern, operator):
    """ Creates a subset based on 1 column and a column value e.g
    column named Names can be subbed to only contain the names 'peter'.
    The rest of the data frame will remain the same.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' is the pandas data frame object.

    column_name : str(optional)
        The parameter called 'column_name' is a string entry for any peculiar
        column found in 'df'.

    pattern : int or float or str
        The parameter called 'pattern' takes any given value found in
        'column_name'.

    operator : str(optional)
        The parameter called 'operator' is string and has 6 valid entries.
        The accepted operator are '==', '!=', '<', '>', '<=' and '>='.

    Returns
    -------
    df_sub : pandas.core.frame.DataFrame(object)
        The 'df_sub' is a subset from df.

    Raises
    ------
    AssertionError
        If 'column_name' string not found in columns of 'df'.

    """
    # Copying frame
    df_work1 = ds.dataset_copy_frame(df)

    # Clearing unnamed columns
    df_work1 = ds.dataset_clear_columns(df_work1)

    # List with columns found!
    columns = df_work1.columns.tolist()

    # Test to see if columns exist

    try:
        assert column_name in columns

    except AssertionError:
        print ('\nThe column name enter is not found\n'
               'Please check entry column_name.')
        return

    # Creating subset frame
    if operator == '==':
        df_sub = df_work1[df_work1[column_name] == pattern]
        return df_sub
    elif operator == '!=':
        df_sub = df_work1[df_work1[column_name] != pattern]
        return df_sub

    elif operator == '>':
        df_sub = df_work1[df_work1[column_name] > pattern]
        return df_sub
    elif operator == '<':
        df_sub = df_work1[df_work1[column_name] < pattern]
        return df_sub

    elif operator == '>=':
        df_sub = df_work1[df_work1[column_name] >= pattern]
        return df_sub

    elif operator == '<=':
        df_sub = df_work1[df_work1[column_name] <= pattern]
        return df_sub
    else:
        print '\nOperator not valid {0}'.format(operator)
        return


def data_assembler_multiple_subsets_via_listname(
        df, column_name, pattern=[], operator='=='):
    """ The function returns subset via list of things and returns a new
    subset.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' is the pandas data frame object.

    column_name : str(object)
        The parameter called 'column_name' is a string entry for any peculiar
        column found in 'df'.

    pattern : list(object)
        The parameter called 'pattern' takes any given value found in
        'column_name' accepts floats, string, and int.

    operator : str(object)
        The parameter called 'operator' is string and has 6 valid entries. The
        accepted operator are '==', '!=', '<', '>', '<=' and '>='.

    Returns
    -------
    df_empty : pandas.core.frame.DataFrame(object)
        The 'df_empty' is a subset from 'df' with multiple values for specific
        column label.

    Raises
    ------
        AssertionError
            If 'column_name' string not found in columns of 'df'.
    """
    # Copying frame
    df_work1 = ds.dataset_copy_frame(df)

    # Creating an empty data frame
    df_empty = pd.DataFrame()

    # Assembling Subset
    for i in pattern:
        temp_file = data_assembler_creating_subset(
            df_work1, column_name, i, operator=operator)
        df_empty = df_empty.append(temp_file)

    return df_empty


def data_assembler_subsetter(
        df, index=[], columns=[], locked_value=[], values=[], index_values=[]):
    """The function takes a concatenated file and divides it according
    to user input. The function returns a new table.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' is the concatenated object.

    index : list(optional)
        The parameter called 'index' specifies how to order the 'df' input. If
        parameter is left empty a function call is trigger to choice index. The
        'index' parameter will order the data by acceding order and by it's
        unique identifiers. The 'index' parameters has length of 1.

    columns : list(optional)
        The parameter called 'columns' specifies how the 'df' input is divided
        or grouped. If 'column' is left empty a function call is trigger and
        user is forced to choice a 'work columns'.

    locked_value : list(optional)
        The parameter called 'locked_value' divides the 'df' according to the
        specific value. The 'locked_value' must be a value present in the
        'columns'.

    values : list(optional)
        The parameter called 'values' is an alternative way of dividing the
        data in 'df'. In contrast to 'locked_value', which split data by a
        repeated value. The 'values' parameter splits the 'df' data by its
        given value and the ascending order which is given by the 'index'
        input.

    index_values : list(optional)
        The 'index_values' parameter works only with 'values' and its a way of
        overriding the given order by 'index'.

    Returns
    -------
    new_frames : list(pandas.core.frame.DataFrame(object))
        The list object contains an -nth amount of subsets derived from the
        original 'df' input.

    Raises
    ------
    ValueError
        If values given in 'index' or 'columns' is not find in 'df'.
        If 'index' contains more than 1 element to order 'df'.
        If 'index_values' is not found in 'df' by given 'index'.

    AssertionError
        If parameter called 'values' contains fewer than unique identifiers for
        given 'index'.

        If length in 'values' differed for given 'index_values'.

    See Also
    --------
    phen_o_links.dataset_pick_columns : For more information about function call
                                      called when 'columns' and 'index' are
                                      left empty.

    phen_o_links.dataset_filesave : To save csv-files from 'new_order' return

    """
    # Local global
    new_frames = []
    tmp = []
    picked = []
    new_frame = []

    # Copying main input
    df_work1 = ds.dataset_copy_frame(df)

    df_columns = df_work1.columns.tolist()

    # Parameter index and column not empty
    if not index or not columns:
        columns, index = ds.dataset_pick_columns(df_work1, split='groupby')

    if index or columns:
        try:
            [df_columns.index(labels) for labels in columns]
            [df_columns.index(order) for order in index]

        except ValueError:
            text = ("\nThe values entered are not found as a column labels in "
                    "'df'. Please reenter values for 'index' and 'columns'. "
                    "Entered values was index = {0} and "
                    " columns = {1} \n").format(index, columns)
            print text
            return
    if len(index) > 1:
        text = ("The 'index' parameter should only contain one element. "
                "The values given by user was {0}. ").format(index)
        raise ValueError(text)

    # Grouping by
    orderby = index + columns
    df_gr = df_work1.groupby(orderby)

    # Getting list of groups
    groups = df_gr.indices.keys()

    # Sorting by index
    groups.sort()

    # Creating a Temporary frame
    tmp_df = pd.DataFrame(groups)

    # Getting index order
    list_of_order = tmp_df[0].unique()

    if locked_value:
        # Creating Slots
        for i in range(len(locked_value)):
            tmp.append([])

        # Adding data to tmp object
        for i in range(len(locked_value)):
            for indicies in list_of_order:
                tmp[i].append((indicies, locked_value[i]))

    if values and not index_values:
        # Checking that indices are equal to order
        try:
            assert len(values) == len(list_of_order)

        except AssertionError:
            text = ("\nThe parameter called 'values' differ from the order "
                    "given by 'index'. The entered input for 'values' is {0} "
                    "and unique identifiers "
                    "in 'index' are {1}\n.").format(values, list_of_order)
            print text
            return

        # Adding data to tmp
        for i in range(len(values)):
            tmp.append((list_of_order[i], values[i]))

    if index_values:
        # Checking that indices are same length
        try:
            assert len(values) == len(index_values)

        except AssertionError:
            text = ("\nThe parameter called 'values' differ from the order "
                    "given by 'index_values'. The entered input for 'values' "
                    "is {0} and unique identifiers in 'index_values' "
                    "are {1}\n.").format(values, index_values)
            print text
            return

        # Check that index_values are present
        try:
            [list_of_order.tolist().index(i) for i in index_values]

        except ValueError:
            text = ("The values in 'index_values' are not present in "
                    "given 'index'. Entered 'index_values' was {0} "
                    "and 'index' values present was "
                    "{1}").format(index_values, list_of_order)
            print text
            return

        # Adding data to tmp
        for i in range(len(values)):
            tmp.append((index_values[i], values[i]))

    # Adding containers
    if locked_value:
        for i in range(len(tmp)):
            picked.append([])

        for i in range(len(tmp)):
            for indicies in tmp[i]:
                picked[i].append(df_gr.get_group(indicies))
        # Ugly but effective
        for i in picked:
            new_frame.append(i)
    else:
        for i in tmp:
            print len(tmp)
            print i
            new_frame.append(df_gr.get_group(i))
        new_frames = pd.concat(new_frame, axis=0)
        return new_frames

    for i in new_frame:
        frame = pd.concat(i, axis=0)
        new_frames.append(frame)

    return new_frames


def data_assembler_reorder(df=[], filepath='', columns=[], delimiter='\t'):
    """ Takes either a path to csv-file or pandas.core.frame.DataFrame(object)
    and re-order the columns according to user input.

    Parameters
    ----------
    df : list(pandas.core.frame.DataFrame(object))
        The parameter called 'df' is list object that accepts only
        pandas data frame objects

    filepath : str(optional)
        The parameter called 'filepath' is the relative path from current
        working directory to a data file. The file path to data file must be
        given as a string. Files format accepted by function is csv-files.

    columns : list(object)
        The parameter called 'columns' are the column labels find either in
        'filepath' or 'df' data. If 'df' parameter contains multiple frames
        it's assumed that the first frames column labels is valid for the nth
        -entry in 'df'. If parameter called 'columns' is left empty it triggers
        function call to determine order.

    delimiter : str(optional)
        The parameter called 'delimiter' determines how file in 'filepath' is
        read. The default values is set to <<TAB>>.

    Returns
    -------
    new_order : pandas.core.frame.DataFrame(object)
        The 'new_order' is either a list object with nth-amount of pandas
        frames or a pandas frame with labels rearrange according to
        the given values in 'columns'.

    Raises
    ------
    AssertionError
        If 'df' parameter inputs are not pandas data frame objects. If
        'filepath' is not a string object.

    ValueError
        If 'filepath' does not point to a file.
        If 'columns' values differed from original list of labels in 'df' or
        'filepath'.

    See Also
    --------
    phen_o_links.dataset_pick_columns : For more information about function call
                                      made when 'columns' empty

    phen_o_links.dataset_filesave : To save csv-files from 'new_order' return

    """
    #Local Global
    df_work1 = []
    columns_labels = []
    new_order = []
    state=0
    # Checking input data

    if df and not filepath:
        try:
            assert(all(isinstance(i, pd.DataFrame) for i in df))

        except AssertionError:
            input_data = [type(i) for i in df]
            text = ("The 'df' parameter accepts only pandas data frames. User "
                    "input was:\n {0}\n").format(input_data)
            print text
            return
        df_work1 = df
        state=1
    elif filepath and not df:
        try:
            assert(isinstance(filepath, basestring))

        except AssertionError:
            text = ("The parameter called 'filepath' only accepts strings! "
                    "User input was typed as {0} for 'filepath' "
                    "parameter. ").format(type(filepath))
        print text
        return
        try:
            if not os.path.isfile(filepath) or not filepath.endswith('.csv'):
                raise ValueError

        except ValueError:
            text = ("The path to file given 'pathfile' does not point to a "
                    "csv-file. User input was {0}  \n .").format(filepath)
        print text
        return

        # Importing the data frame
        df_work1 = pd.read_csv(filepath, delimiter=delimiter, header=0)

    # Check column labels in original set
    if len(df_work1) > 1 or state>0 :
        columns_labels = df_work1[0].columns.tolist()
    else:
        columns_labels = df_work1.columns.tolist()

    if not columns:
        frame = pd.DataFrame(columns=columns_labels)
        columns, index = ds.dataset_pick_columns(frame, split='groupby')
        del index

    if columns:
        # Check that values in columns exists
        try:
            [columns_labels.index(i) for i in columns]

        except ValueError:
            text = ("Values in 'columns' dose not match column labels in input"
                    " data. User input was \n {0} \n and found "
                    "labels were:\n {1} .").format(columns, columns_labels)
            print text
            return

    # Fixing order with 'columns'

    if len(df_work1) > 1 or state > 0:
        new_order = [i.reindex(columns=columns) for i in df_work1]
    else:
        new_order = df_work1.reindex(columns=columns)

    return new_order


def data_assembler_unique_keys(df=[]):
    """Takes any nth- amount pandas.core.frame.DataFrame(object) and creates
    a unique identifiers is used for assembling different data sets!
    The unique identifiers is based on the length of the data frame input and
    is therefore limited to input of equal length to work properly.
    Use after assembled data has been reorder.

    Parameters
    ----------
    df : list(pandas.core.frame.DataFrame(object))
        The parameter called 'df' is list object that only accepts pandas.core.
        frame.DataFrame objects.

    Returns
    -------
    df : list(pandas.core.frame.DataFrame(object))
        The returns the frames in input 'df' with new column called
        'Unique_id'.

    Raises
    ------
    ValueError
        If 'df' inputs is not a pandas.core.frame.DataFrame(object)

    See Also
    --------
    data_assembler_reorder : For more information about how to reorder frames.

    data_assembler_II_multiple_files_import_CSV_format : For more information
                                                         about how to import
                                                         data from csv-file.

    data_assembler_subsetter : For more information about how to split data.

    phen_o_links.dataset_flag_remover : To format 'ORF'- names properly

    Notes
    -----
    The function should be called after above mentioned functions!
    A complementary function called 'data_assembler_global_uniquekeys' is
    advised to run after.

    """
    try:
        if not all(isinstance(i, pd.DataFrame) for i in df):
            raise ValueError

    except ValueError:
        object_types = [type(i) for i in df]
        text = ("The 'df' parameter should only contain {0} and the parameter "
                " contained: {1} "
                "types.").format(type(pd.DataFrame()), object_types)
        print text
        return

    for i in df:
        i.insert(0, 'Unique_Id', value=[
            'A_' + str(label) for label in range(len(i))])

    return df


def data_assembler_global_uniquekeys(df, column=''):
    """The function creates a dictionary object where the value are the unique
    keys give by the function call called 'data_assembler_unique_keys'.


    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The 'df' parameter is a pandas frame with a column labelled as
        'Unique_Id.

    column : str(optional)
        The 'column' is a column labelled present in 'df'. The 'column'
        function as the key.

    Returns
    -------
    global_keys : dict(object)
        The return value called 'global_keys' is dictionary where values
        are the unique keys given by 'data_assembler_unique_keys'.

    csv-file : file
        A csv-file called 'global_identifiers' with items form 'global_keys'.
        File is created in current working directory.

    Raises
    ------
    ValueError
        If 'df' does not contain a column labelled as 'Unique_Id' or if
        'column' as well is missing in 'df'.

    See Also
    --------
    phen_o_links.dataset_to_dict : For more information about 'global_key'.

    phen_o_links.dataset_flag_remover : For orf name formatting.

    phen_o_links.dataset_add_column_by_dict : For more information on how to
                                            'global_key' to pandas data frames.

    Notes
    -----
    The orf names should be formatted as the following string 'YAL002W'.

    """
    # Local global
    var1 = "Unique_Id"
    var2 = "global_identifiers"
    # Copy frame
    df_work1 = ds.dataset_copy_frame(df)
    column_labels = df_work1.columns.tolist()

    # Checking for column labels
    try:
        column_labels.index(var1)
        column_labels.index(column)

    except ValueError:
        text = ("The pandas frame called 'df' has no labelled "
                "called 'Unique_Id'. Found columns in 'df' "
                "was: \n {0}").format(column_labels)
        print text
        return
    # Creating global keys
    global_keys = ds.dataset_to_dict(df_work1, keys=column, values=var1)

    # Creating pandas frame
    keys_file = pd.DataFrame(data=global_keys.items(), columns=[column, var1])
    ds.dataset_filesave(keys_file, filename=var2, delimiter='\t')

    return global_keys


def data_assembler_merge_frames(df=[], key=''):
    """The function merges 2 pandas data frames to one pandas data frame with
    help of the unique identifiers that are given with
    data_assembler_unique_keys function.

    Parameters
    ----------
    df : list(pandas.core.frame.DataFrame(object))
        The parameter called 'df' accepts only pandas data frames as valid
        entries. The parameter must contain 2 entries.

    key : str(optional)
        The parameter called 'key' accepts string input. The input in 'key'
        must be a present column labelled for both pandas frames. If 'key'
        parameter is left empty triggers function call.

    Returns
    -------
    df_merged_nans, df_merged : pandas.core.frame.DataFrame(object)
        The return values 'df_merged_nans' and 'df_merged' are pandas data
        frame. The 'df_merged_nans' is merged with the how set to 'outer',
        whereas the df_merged is set to 'inner'.
    csv : file
        The merged frames are also saved at current working directory!
        Files named 'merged_frames_with_nans' for ' outer' merged frames and
        'merged_frames' for 'inner' merged frames.

    Raises
    ------
    ValueError
        If 'df' inputs are not pandas data frames and if 'key' input is not
        found in 'df' twice.

    IndexError
        If 'df' input is not equal to 2.

    See Also
    --------
    phen_o_links.dataset_filesave : For more information about function called
                                  when is empty 'key'.


    """

    # Checking user inputs
    try:
        if not all(isinstance(i, pd.DataFrame) for i in df):
            raise ValueError
    except ValueError:
        typed_df = [type(i) for i in df]
        text = ("The entries in given in 'df' are not pandas data frame "
                "objects. User input was {0}.").format(typed_df)
        print text
        return

    try:
        if not len(df) == 2:
            raise IndexError
    except IndexError:
        text = ("The 'df' contains more than 2 elements! "
                "User input was {0}.").format(len(df))
        print text
        return

    column_labels = df[0].columns.tolist() + df[1].columns.tolist()

    if not key:
        text = ("Displaying first item in 'df' pick 1 column label that is "
                "present in both data frames for the given data input in "
                "'df'\n")
        print text

        work, index = ds.dataset_pick_columns(df[0], split='groupby')
        key = str(work[0])

    try:
        if not column_labels.count(key) == 2:
            raise ValueError

    except ValueError:
        text = ("The entry made for 'key' is not found in both frames given "
                "in 'df' parameter. Columns found {0} and 'key' entries "
                " was {1}").format(column_labels, key)
        print text
        return
    # Checking indices present
    df_nr = [len(i) for i in df]

    # Order equal 1 change order
    order = df_nr.index(np.max(df_nr))
    if order == 1:
        df = df[::-1]

    # Merging frames
    df_merged_nans = df[0].merge(df[1], how='outer', on=key)
    df_merged = df[0].merge(df[1], how='inner', on=key)

    print "Saving frames!"
    ds.dataset_filesave(df_merged_nans, filename='merged_frames_with_nans')
    ds.dataset_filesave(df_merged, filename='merged_frames')
    return df_merged_nans, df_merged


def data_assembler_import_interaction_db(
        filepath='',delimiter='\t',filter_type=[],row_skips=9):
    """The function imports the SGD interaction database for a singular query
    gene (i.e. the background gene in the SGA-collection).

    Parameters
    ----------
    filepath : str(object)
    The 'filepath' is the relative or absolute path to the data base file.
    Accepts only string entries.

    delimiter : str('object')
    The 'delimiter' parameter determines how the data base file should be
    parsed. Default value for 'delimiter' is <<Tab>> separated.

    filter_type : list(object)
    The 'filter_type' object list object that only accepts string entries.
    The interaction database has 2 types of interaction either 'Genetic' or
    'Physical'. Default value for 'filter_type' is null.

    row_skips : int(object)

    """

    #Local Global
    types = ["Genetic", "Physical"]
    column_names = ['Gene_Name', 'Systematic_Name', 'Gene_Name_interactor',
                    'Interactor_Systematic_Name', 'Type', 'Assay',
                    'Annotation', 'Action', 'Modification', 'Phenotype',
                    'Source', 'Reference', 'Note']


    # Checking that parameters are good
    if not isinstance(row_skips, int):
        text = ("Parameter 'row_skips' accepts only integers:"
                "\n {0}").format(row_skips)
    if not isinstance(filepath, str):
        text = ("The parameter 'filepath' is not a"
                "string type:\n {0}").format(type(filepath))
        raise TypeError(text)

    if not os.path.isfile(filepath):
        text = ("Path given to 'filepath' does not point to a file")
        raise OSError(text)

    if not all(isinstance(i, str) for i in filter_type):
        text = ("The parameter 'filter_type' contains other type "
                "than string:\n{0}.").format(filter_type)
        raise TypeError(text)

    if not all([i in types for i in filter_type]):
        text = ("An invalid entry was given for 'filter_type':\n{0}\n "
                "Please enter either 'Genetic' or "
                "'Phsyical'.").format(filter_type)
        raise ValueError(text)

    # Importing database
    interaction_db = pd.read_table(filepath, header=None, sep=delimiter,
                                   skiprows=row_skips)

    # Relabel columns
    if interaction_db.shape[1] == len(column_names):
        re_columns = {i: column_names[i] for i in range(len(column_names))}
        interaction_db = interaction_db.rename(columns=re_columns)
        interaction_db = interaction_db[interaction_db['Type'] == filter_type[0]]
        return interaction_db

    else:
        interaction_db = interaction_db[interaction_db[4] == filter_type]
        return interaction_db


if __name__ == "__main__":
    # Execute only as script
    print "Please import module named {0} with Ipython".format(__name__)
