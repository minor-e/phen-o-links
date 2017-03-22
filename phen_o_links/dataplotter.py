#!/usr/bin/env python

#Data plotter script created by Esteban Fernandez Parada
#Date 22-07-2014
#Alpha developed no data refund for lost data!
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as distance
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize
from os import listdir
import dataset as ds

# Globals
# Dict object only used for go enrichment plots!
go_colors = {}

# Classes

class MidpointNormalize(Normalize):
    """
    Color normalizes heat map to a midpoint value. The value the zero
    value will be blank in the heat map. See web page for source code `Web`_
    and the reference for `author`_ of code.

    Attributes
    ----------
    None

    Methods
    -------
    None

    References
    ----------

    .. [1] Code was copied from Joe Kington, date:10-09-2015, time: 11:00. For
           more information about Joe Kington `author`_ click link.

    .. _author: http://stackoverflow.com/users/325565/joe-kington

    .. _Web: http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib

    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Helper functions


def dataplotter_dendrogram_maker(
        data, plot=False, pos='top', cutoff=None):
    """Function returns a dictionary typed dendrogram from data input.

    Parameters
    ----------
    data : numpy.ndarray(object)
        The parameter called 'data' is data that has been
        pre-clustered by scipy.cluster.hierarchy.

    plot : boolean(optional)
        The parameter 'plot' renders either a figure
        of the dendrogram or not. Default value for 'plot' is 'False', which
        renders dendrogram figure.

    pos : str(optional)
        The parameter called 'pos' is string typed and it roots the dendrogram
        according to given orientation. The default value of
        'pos' is 'top'. Valid entries are 'top', 'bottom', 'left', 'right'.

    Returns
    -------
    dendrogram : dict(object)
        The return values called 'dendrogram' is dictionary typed object.
        Key values in dictionary are 'ivl', 'dcoord', 'leaves', 'color_list'
        and 'icoord', with the correspondent values to be string, float, int,
        string, float.

    Raises
    ------
    ValueError
        If string entry for 'pos' is not valid option.

    IndexError
        If 'color' parameter is empty.

    TypeError
        If 'color' object not list object or If 'data' object is not a
        numpy.ndarray object.
    """
    # Global local variable
    root_orientation = ['top', 'bottom', 'left', 'right']

    # Checking that given inputs are correct.
    try:
        if pos not in root_orientation:
            raise ValueError
    except ValueError:
        print ("The string in 'pos' parameter not valid, "
               "here are valid options:\n {0}").format(root_orientation)
        return
    #try:
    #    if not color:
    #        raise IndexError
    #except IndexError:
    #    print ("Parameter called 'color' is empty:\n {0}").format(color)
    #    return

    #try:
    #    if not isinstance(color, list):
    #        raise TypeError
    #except TypeError:
    #    print ("Parameter called 'color' "
    #           " is not list type:\n {0}").format(type(color))
    try:
        if not isinstance(data, np.ndarray):
            raise TypeError

    except TypeError:
        print ("Parameter called 'data' "
               " is not numpy.ndarray type:\n {0}").format(type(data))


    # Making dendrogram plot
    dendrogram = sch.dendrogram(
        data, no_plot=plot, orientation=pos, color_threshold=cutoff)

    # Return
    return dendrogram


def dataplotter_table_fontmanger(fontsize, maxlengthword):
    """Handles font for tables and calculates width and height from
    font size given.

    Parameters
    ----------
    fontsize : int or float
        The parameter called 'fontsize' is the given font size to scale from.
        The parameter works with font sizes >= 12.
    maxlengthword : int
        The parameter called 'maxlengthword' is the character length of the
        longest possible word in table.

    Returns
    -------
    cell_width, cell_height : float
        The parameters called 'cell_width' and 'cell_height' are the cell
        height and cell width for each individual cell in table.

    Raises
    ------
    ZeroDivisionError
        If 'fontsize' parameter is < 12
    """

    # Cell Height Linear Equation
    c_height = 0.0075
    m_height = 0
    cell_height = (c_height * fontsize) + m_height

    # Checking font size

    try:
        12 / ((int(fontsize) / 12) * 12)
    except ZeroDivisionError:
        print '\nFont size must be => 12'
        return

    # Cell Width Linear Equation
    c_width = -6.217248937900874e-15
    cell_scale = (fontsize / 12)
    m_width = 5.555555555555562
    x = (maxlengthword * 1.0) / fontsize
    cell_width = (((c_width * x) + m_width) * cell_height) * cell_scale

    return cell_width, cell_height


def dataplotter_colorscheme(main=[], hues=[]):
    """Functions returns multiple list with basic color schemes and hues.

    Parameters
    ----------
    main : list(optional)
        The parameter called 'main' determines the foundation color of the
        color scheme. Valid entries as string in list are 'green', 'red',
        'blue', 'yellow', 'black', 'white', 'colorblind', grayscale and
        grayscale2.

    hues : list(optional)
        The parameter called 'hues' is list and takes only string entries. The
        parameter determines hues for picked foundation color. Valid string
        entries are 'light', 'lightdark', 'darklight' and 'dark'.

    Returns
    -------
    maincolors, light, light_dark, dark_light, dark : list(object)
                                                    Returned parameters are
                                                    order by main color for
                                                    each corresponding hue.

    Examples
    --------
    >>> main, light, light_dark, dark_light, dark = dataplotter_colorscheme(
            main=['green'], hues=['light'])
    >>> # Colors returned
    >>> main[0] = 'green'
    >>> light[0] = 'light green'

    """
    # Local global variables
    light = []
    light_dark = []
    dark_light = []
    dark = []

    # Super nested colour
    colorscheme = {
        'green': {'main': '#37B209', 'light': '#A0E388',
                  'lightdark': '#78E351',
                  'darklight': '#3E6A2E', 'dark': '#154D01'},
        'red': {'main': '#C50A28', 'light': '#EA8C9B', 'lightdark': '#EA536B',
                'darklight': '#75333D', 'dark': '#55010E'},
        'blue': {'main': '#0C5F83', 'light': '#82BAD2', 'lightdark': '#51ABD2',
                 'darklight': '#24414E', 'dark': '#012838'},
        'yellow': {'main': '#D0730B', 'light': '#EEC18E',
                   'lightdark': '#EA6A55',
                   'darklight': '#7C5B36', 'dark': '#5A3001'},
        'black': {'main': '#121218', 'darklight': '#0B0B0D',
                  'dark': '#010008'},
        'white': {'main': '#ffffff', 'light': '#F1F0F8',
                  'lightdark': '#DBD8F8'},
        'colorblind': {'main': '#3A487A', 'light': '#8892B4',
                       'lightdark': '#5C6995', 'darklight': '#223161',
                       'dark': '#0D1A44'},
        'grayscale': {'main': '#8B8B8B', 'light': '#E8E8E8',
                      'lightdark': '#B7B7B7', 'darklight': '#5C5C5C',
                      'dark': '#1D1D1D'},
        'grayscale2': {'main': '#636363', 'light': '#A3A3A3',
                       'lightdark': '#7E7E7E', 'darklight': '#494949',
                       'dark': '#292A2A'}}
    # Appending colours
    maincolors = [colorscheme[main[i]].get('main') for i in range(len(main))]
    if hues.count('light') > 0:
        light = [colorscheme[main[i]].get('light') for i in range(len(main))]
    if hues.count('lightdark') > 0:
        light_dark = [colorscheme[main[i]].get('lightdark')
                      for i in range(len(main))]
    if hues.count('darklight') > 0:
        dark_light = [colorscheme[main[i]].get('darklight')
                      for i in range(len(main))]
    if hues.count('dark') > 0:
        dark = [colorscheme[main[i]].get('dark')
                for i in range(len(main))]
    # Returning colours
    return maincolors, light, light_dark, dark_light, dark


def dataplotter_object_table_maker(
        df, cell_height=0.09, figtext=['Untitled'], ftsz=12):
    """Converts pandas objects e.g. data frames to figure objects.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        Input frame to convert from object to figure.

    cell_height : float or int (optional)
        The parameter called 'cell_height' determines
        the cell height for each individual cell in table.

    figtext : list(optional)
        The parameters 'figtext' gives the title to rendered table.

    ftsz : float or int (optional)
        The parameter called 'ftsz' determines font size in use
        for rendered table.

    Returns
    -------
    fig1 , ax1 : matplotlib.figure.Figure, matplotlib.axes.AxesSubplot
        The function returns 'fig1' and 'ax1' both matplotlib
        objects. The 'fig1' return contains the converted table.

    Raises
    ------
    NameError
        If user interaction dialog input is erroneous.

    See Also
    --------
    dataplotter_save_figure : "fig1" return is used for save.

    dataplotter_fixing_textformat : For more information about
                                    latex rendering

    phen_o_links.dataset.dataset_copy_frame : For more information about
                                            data frame copy.

    phen_o_links.dataset.dataset_pick_columns : For more information
                                              about "split" parameter.
    """
    # Copying frame
    df_work1 = ds.dataset_copy_frame(df)

    df_work2, indexby = ds.dataset_pick_columns(df_work1, split='indexby')

    # Getting column names and row names
    colnames = df_work2.columns.tolist()
    rownames = df_work2.index.tolist()
    colnamesmax = np.max([len(i) for i in colnames])

    # Fixing text issues
    colnames = dataplotter_textspacemanger(colnames)
    rownames = dataplotter_textspacemanger(rownames)
    tableTitle = dataplotter_textspacemanger(figtext)

    # Getting data
    table_values = df_work2.values

    # Creating Figure
    dataplotter_fixing_textformat()

    fig1 = plt.figure(figsize=(4, 3))
    ax1 = fig1.add_subplot(111, aspect='equal')

    table = ax1.table(
        cellText=table_values,
        colLabels=[r'\textbf{%s}' % (colnames[i]) for i in range(
            len(colnames))],
        rowLabels=[r'\textmd{%s}' % (rownames[i]) for i in range(
            len(rownames))],
        loc='best')

    inputuser = raw_input('Creating table directly from csv-file?(Y/N) \n:')
    inputuser = inputuser.lower()

    try:
        if inputuser == 'y' or inputuser == 'n':
            print 'User input accepted'
        else:
            raise NameError('User input must be either "Y" or "N"')
    except NameError:
        print 'User input must be either "Y" or "N"'
        return

    if inputuser == 'y':
        ftsz = 44 * (44 / 24.0)
        c_w, c_h = dataplotter_table_fontmanger(32, colnamesmax)
        cell_height = c_h
        cell_width = c_w
        table.auto_set_font_size(False)
        table.set_fontsize(32)
        table.scale(10, 10)

    table_props = table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_height(cell_height)
        if inputuser == 'y':
            cell.set_width(cell_width)

    ax1.set_title(r'\textbf{%s}' % (tableTitle[0]), fontsize=ftsz,
                  loc='center')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    return fig1, ax1


def dataplotter_stats_table_maker(
        df, filter_val, cell_height=0.09,
        pattern=' ', split='groupby', figtext=['Untitled'], ftsz=12):
    """Takes 2 columns and returns a table with statistics.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        Input data frame.

    filter_val : int, float (optional)
        Value for filtering columns.

    cell_height : int, float (optional)
        The height for each given cell in table.
        The default value for "cell_height" is 0.09.

    pattern : str (optional)
        The columns name formatting for latex. Parameter used for
        str.replace substitution on column names.
        The default value for "pattern" is "_".

    split : str (optional)
        Parameter used for returning list of columns.
        The default value for "split" is "groupby".

    Returns
    -------
    fig2 : plt.figure.Figure (object)
        Matplotlib.pyplot (plt) figure object.
        Renders figure or in this case table.

    table2 : pd.DataFrame (object)
        Pandas data frame object with values and
        column names.

    table3 : plt.table (object)
        The subplot in figure as table object.

    See Also
    --------
    dataplotter_save_figure : "fig2" return is used for save.

    dataplotter_fixing_textformat : For more information about
                                    latex rendering

    phen_o_links.dataset.dataset_copy_frame : For more information about
                                             data frame copy.

    phen_o_links.dataset.dataset_filter_by_value : For more information
                                                  about "filter_val" parameter.
    """
    # Fixing text
    dataplotter_fixing_textformat()

    # Copying data frame
    df_work1 = ds.dataset_copy_frame(df)

    # Filtering and to given subsets
    df_all, df_work2, x, y = ds.dataset_filter_within_a_range_of_value(
        df_work1, filter_val, split=split)

    # Creating figure and axis
    fig2 = plt.figure(figsize=(4, 3))
    ax2 = fig2.add_subplot(111, aspect='equal')

    # Creating Table names
    x_title = [str(x.name)]
    y_title = [str(y.name)]

    # Fixing titles names
    x_title = dataplotter_textspacemanger(x_title)
    y_title = dataplotter_textspacemanger(y_title)
    tableTitle = dataplotter_textspacemanger(figtext)

    # Labels for columns
    col_labels = [r'\textbf{%s}' % (x_title[0]),
                  r'\textbf{%s}' % (y_title[0])]
    row_labels = ['Mean', '$s^2$', 's', '25\%',
                  '50\%', '75\%',
                  'Frequency\ Lower', 'Frequency\ Upper']

    # Statistic calculation for table
    x_stat = [np.mean(x), np.var(x, ddof=1), np.std(x, ddof=1)]
    x_per = np.percentile(x.values, [25, 50, 75])

    x_freq = [sum([i for i in x <= x_per[0] if i == True]),
              sum([i for i in x >= x_per[2] if i == True])]

    y_stat = [np.mean(y), np.var(y, ddof=1), np.std(y, ddof=1)]
    y_per = np.percentile(y.values, [25, 50, 75])
    y_freq = [sum([i for i in y <= y_per[0] if i == True]),
              sum([i for i in y >= y_per[2] if i == True])]

    # Putting together statistical data
    x_stat = np.append(x_stat, x_per, axis=1)
    x_stat = np.append(x_stat, x_freq, axis=1)
    y_stat = np.append(y_stat, y_per, axis=1)
    y_stat = np.append(y_stat, y_freq, axis=1)

    # Creating numpy arrays of stats
    x_stats = np.asarray(x_stat)
    y_stats = np.asarray(y_stat)

    # Putting together creating table from arrays
    table = np.insert(y_stats,
                      np.arange(len(x_stats)), x_stats)

    table = table.reshape(8, 2)

    # Creating a csv version of table
    table2 = pd.DataFrame(data=table,
                          columns=[col_labels[i]
                                   for i in range(len(col_labels))],
                          index=row_labels)
    # Table making
    table3 = ax2.table(cellText=table,
                       rowLabels=[r'\textit{%s}' % (row_labels[i])
                                  for i in range(len(row_labels))],
                       colLabels=[col_labels[i]
                                  for i in range(len(col_labels))],
                       loc='best')
    table_props = table3.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells:
        cell.set_height(cell_height)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_frame_on(False)
    ax2.set_title(r'\textbf{%s}' % (tableTitle[0]), fontsize=ftsz)
    #plt.show()

    return fig2, table2, table3


def dataplotter_save_figure(
        fig, filename='Untitled', path='./',
        save_as=["pdf", "svg", "png", "eps"], legend_tight=[False],
        axes_container=[]):
    """The function saves displayed figures. The figure
    size is set to 16,12 for figures. For tables it returns the same size.

    Parameters
    ----------
    fig : plt.figure.Figure (object)
        Figure wanted to be saved.

    filename : str (optional)
        File name for figure. The default value
        for "filename" is 'Untitled'.

    path : str (optional)
        The file path where file for figure is saved at.
        The default value for "path" is "./", which is the current
        working directory.

    save_as : list optional
        The suffix and format of the save file.
        The default value for "save_as" are pdf,
        svg, png and eps.

    axes_container : list(object)
        The parameter called 'axes_container' is list object with length
        1 item that is obligated to be  matplotlib.axes.AxesSubplot object!

    legend_tight : list(optional)
        The parameter called 'legend_tight' has length of 1 item
        and the item is boolean object. If parameter is set to True
        a legend is incorporated at right side of the rendered
        figure.

    Raises
    ------
    OSError
        If "path" parameter can't be listed.

    AttributeError
        If "fig" parameter can't extend window.
    AssertionError
        if item in 'axes_container' is not Matplotlib subplot axes object.

    Returns
    -------
        Figure saved in multiple formats with a
        fix figure size of (16 width, 12 height).
    """
    try:
        listdir(path)
    except OSError:
        print "Directory name not found in system!"
        return

    try:
        fig.get_window_extent()

    except AttributeError:
        print "Not a figure object from Matplotlib."
        return

    # Returns width and height of figure
    window_mng = fig.get_size_inches()

    # Check spelling of name variable
    if filename.endswith('.'):
        filename = filename[:-1]

    # Setting window size and set to full resolution.
    if 4 < window_mng[0] < 16:
        w, h = window_mng[0] * 2, window_mng[1] * 2
        fig.set_size_inches(w, h)
        # Removes frame of figure
        fig.set_frameon(False)
        # Saving figures with multiple formats.
        for i in range(len(save_as)):
            fig.savefig(path + filename + str(i) + "." + save_as[i],
                        format=save_as[i], bbox_inches="tight")

    if legend_tight[0]:
        try:
            assert type(plt.axes()) == type(axes_container[0])

        except AssertionError:
            print ("Item in 'axes_container' is not"
                   " a matplotlib.axes.AxesSubplot object."
                   "\nPlease enter a correct item in axes_container!")
            return
        # Creating handle object for legend
        handles, labels = axes_container[0].get_legend_handles_labels()
        legend = axes_container[0].legend(
            handles, labels, loc='upper left', frameon=False,
            bbox_to_anchor=(1.01, 0.95), borderaxespad=0.)
        for i in range(len(save_as)):
            fig.savefig(path + filename + str(i) + "." + save_as[i],
                        format=save_as[i], bbox_inches="tight",
                        bbox_extra_artist=(legend, ))

    else:
        # Setting window size
        w, h = window_mng[0], window_mng[1]
        fig.set_size_inches(w, h)
        # Removes frame of figure
        fig.set_frameon(False)
        # Saving figures with multiple formats.
        for i in range(len(save_as)):
            fig.savefig(path + filename + str(i) + "." + save_as[i],
                        format=save_as[i], bbox_inches="tight")


def dataplotter_x_y_limit(
    PlotAxess, x_mini=0, y_mini=0,
        x_limit=10, y_limit=10):
    """Sets x and y limit within a figure.

    Parameters
    ----------
    PlotAxess : matplotlib.axes_subplots.AxesSubplot
        The parameter handles axes and subplot of figure.

    x_mini, y_mini : int, float (optional)
        The x and y axis sub limit for a figure.
        The default "x_mini" value is 0 and default value for
        "y_mini" is 0.

    x_limit, y_limit : int, float (optional)
        The x and y upper limit for a figure.
        The default value for "x_limit" is 10 and default value for
        "y_limit" is 10.

    Returns
    -------
    Figure with x and y axis limitations.
    """
    # Setting window limits
    PlotAxess.set_xlim(x_mini, x_limit)
    PlotAxess.set_ylim(y_mini, y_limit)


def dataplotter_x_y_tick_remover(PlotAxess, all_ticks=False, x=False):
    """Sets ticks to bottom of figure and left side of figure. Only valid for
    2 dimensional plot. The function also removes all ticks via 'all' parameter

    Parameters
    ----------
    PlotAxess : matplotlib.axes_subplots.AxesSubplot
        The parameter handles axes and subplot of figure.
    all : boolean(optional)
        The 'all' parameter removes all ticks from a rendered figure
        if parameter is set to 'True'.

    Returns
    -------
    Figure with axis tick on bottom and left side of figure.
    """
    # Set tick to bottom and left side and bottom side of figure
    PlotAxess.get_xaxis().tick_bottom()
    PlotAxess.get_yaxis().tick_left()

    # Removes all ticks in figure
    if all_ticks:
        PlotAxess.get_xaxis().set_ticks([])
        PlotAxess.get_yaxis().set_ticks([])
    if x:
        PlotAxess.get_yaxis().set_ticks([])


def dataplotter_spines_remover(
        PlotAxess, top=True, bottom=True,
        right=True, left=True, all_axis=True):
    """Removes spines from a figure.

    Parameters
    ----------
    PlotAxess : matplotlib.axes_subplots.AxesSubplot
        The parameter handles axes and subplot of figure.

    top : boolean (optional)
        Handles top spines. The default value for "top" is True.
        If "top" value changed to False "top" spines are not visible.

    bottom : boolean (optional)
        Handles bottom spines. The default value for "bottom" is True.
        If "bottom" value changed to False "bottom" spines are not visible.

    right : boolean (optional)
        Handles spines located to the right side of figure.
        The default value for "right" is True. If "right" value
        changed to False "right" spines are not visible.

    left : boolean (optional)
        Handles spines located to the left side of figure.
        The default value for "left" is True. If "left" value
        changed to False "left" spines are not visible.

    all_axis : boolean (optional)
        Handles all spines at once of figure. The default value
        for "all_axis" is True. This renders a figure without spines.
        If "all_axis" is set to False inputs from parameters top,
        bottom, right and left are valid.

    Returns
    -------
    Figure with or without spines according to directional or "all_axis"
    parameters.
    """
    if all_axis:
        top = False
        bottom = False
        right = False
        left = False
        PlotAxess.spines['top'].set_visible(top)
        PlotAxess.spines['bottom'].set_visible(bottom)
        PlotAxess.spines['right'].set_visible(right)
        PlotAxess.spines['left'].set_visible(left)
    else:
        PlotAxess.spines['top'].set_visible(top)
        PlotAxess.spines['bottom'].set_visible(bottom)
        PlotAxess.spines['right'].set_visible(right)
        PlotAxess.spines['left'].set_visible(left)


def dataplotter_text_annotation_scatter(
    PlotAxess, df_work1, fontsize,
        picked_columns, indexer, color='#DBD8F8'):
    """The function returns a lines with with text annotations for
    outliers groups.

    Parameters
    ----------
    PlotAxess : matplotlib.axes.AxesSubplot (object)
        The 'PlotAxess' is the add subplot for a figure.

    df_work1 : pandas.DataFrame(object)
        Input object which consist of a subset from
        original data frame e.g. outliers or other
        types of small but defined amount of set of values.

    picked_columns : list
        The parameter called 'picked_columns' is a list type object
        that contains string items with 2 column names from
        'df_work1'

    indexer : list
        The parameter called 'indexer' is a list type object
        that contains a column label a string item, which is
        present in 'df_work1'

    fontsize : int or float
        The 'fontsize' determines the font size used in text
        annotation for outliers

    color : str(optional)
        The parameter called 'color' decides the face color of the
        annotation background color. The 'color' parameter
        is set to '#DBD8F8' a grayish color.

    Returns
    -------
    Works directly upon subplot, therefor no need to return output.

    See Also
    --------

    phen_o_links.dataset.dataset_pairwise_distance_points : For more information
                                                           about 'd' variable.

    """
    # Local variables
    bbox_args = dict(boxstyle='round,pad=0.5', fc=color)
    arrow_args = dict(arrowstyle='-', connectionstyle='arc3,rad=0',
                      color='black')
    columns_table = df_work1.columns.tolist()

    # Try and test
    try:
        test = [columns_table[columns_table.index(i)] for i in picked_columns]

    except ValueError:
        print ("\nColumns in data frame does not match with picked columns"
               " Columns present in table:\n {0},"
               " Columns picked:\n {1}").format(columns_table, picked_columns)
        return

    # Slicing subset.
    columns_slice = indexer + picked_columns
    df_work1 = df_work1[columns_slice]

    # Calls function
    d = ds.dataset_pairwise_distance_points(df_work1, picked_columns)

    labels = df_work1[indexer[0]].values
    for i, txt in enumerate(labels):
        x = range(-40, -20, 7) + range(20, 40, 7)
        y = range(-40, -20, 3) + range(20, 41, 5)
        if d[i][0] < 5:
            x = np.random.choice(x)
            y = np.random.choice(y)
        else:
            x = -20
            y = 20
        PlotAxess.annotate(r'%s' % (txt), (
            df_work1[picked_columns[0]].values[i],
            df_work1[picked_columns[1]].values[i]), fontsize=fontsize,
            textcoords='offset points', xytext=(x, y), color='w',
            bbox=bbox_args, arrowprops=arrow_args)


def dataplotter_text_annotation_hist(
        df_sub, patches, bins, colorscheme):
    """Takes any given histogram plot and text annotates according
    to a given subset.

    Parameters
    ----------
    df_sub : pandas.core.frame.DataFrame(object)
        The parameter called 'df_sub' is a subset of larger data frame
        extracted from outliers function or other sort of sub-sampling.

    patches : matplotlib.cbook.silent_list
        The parameter called 'patches' is the return value from
        the histogram function in Matplotlib.

    bins : numpy.ndarray
        The parameter called 'bins' is a return value from histogram
        function in Matpltolib.

    colorscheme : list
        The parameter called 'colorscheme' is a list object with
        items typed as string. Values stem from dataplotter_colorscheme
        function.

    Returns
    -------
    Nothing, however acts upon figure making.
    """
    df_sub['color'] = [colorscheme[i] for i in range(len(df_sub))]
    df_sub2 = ds.dataset_creating_patches_indices(
        df_sub, bins)
    df_sub3, picked_column = ds.dataset_check_duplicated_patches_indices(
        df_sub2)
    for i in range(len(df_sub3)):
        names = str(df_sub3[picked_column[0]].values[i])
        indexes = int(df_sub3['patches'].values[i] - 1)
        colors = str(df_sub3['color'].values[i])
        if indexes < 0:
            indexes = 0
            patches[indexes].set_label(names)
            patches[indexes].set_facecolor(colors)
        else:
            patches[indexes].set_label(names)
            patches[indexes].set_facecolor(colors)


def dataplotter_fixing_textformat():
    """Fixes font and text formatting for figures.

    Parameters
    ----------
    No parameters needed!

    Returns
    -------
    Figure text formated by various latex formats.
    """
    # Text fixing
    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': "sans-serif"})
    params = {'text.latex.preamble': [r'\usepackage{siunitx}',
                                      r'\usepackage{sfmath}',
                                      r'\sisetup{detect-all}',
                                      r'\usepackage{amsmath}']}

    plt.rcParams.update(params)
    return "Text input has been formatted in params"


def dataplotter_textspacemanger(text_list, pattern=' ', output="\ "):
    """Latex space converter format for figure text!

    Parameters
    ----------

    text_list : list (optional)
        The parameter 'text_list' is list that
        contains string elements e.g figure title.

    Returns
    -------
    text_latex_space : list
        The 'text_latex_space' is the same list as 'text_list'
        with the crucial difference that it renders latex
        style spacing.
    """
    # Formatting string to latex spacing.
    text_latex_space = [
        text_list[i].replace(pattern, output)
        for i in range(len(text_list))]
    return text_latex_space


def dataplotter_customlegends(
        PlotAxess, labels=[], colors=[], fontsize=[], frameon=False,
        loc=1, numpoints=1,
        boolean_handlers={'circle': None, 'lines': None, 'rect': None},
        labels_index={'circle': [], 'lines': [], 'rect': []},
        color_index={'circle': [], 'lines': [], 'rect': []},
        circle_art={'x': 0, 'y': 0, 'marker': ['o'], 'linestyle': 'none',
                    'alpha': None},
        lines_art={'x': 0, 'y': 0, 'linestyle': ['--'], 'marker': 'None',
                   'alpha': None, 'linewidth': None},
        rect_art={'x': 0, 'y': 0, 'width': 1, 'height': 1, 'hatch': None,
                  'alpha': None}):
    """Functions return custom legend via dictionary entry.

    Parameters
    ----------
    PlotAxess : matplotlib.axes.AxesSubplot (object)
        The parameter called 'PlotAxess' is matplotlib object.

    labels : list(optional)
        The parameter called 'labels' is list with items types string
        for legend.

    colors : list(optional)
        The parameter called 'colors ' is a list with items
        types string that decides colors for made legends.

    fontsize : list(optional)
        The parameter called 'fontsize' is list with a digit
        for a given font size in points.

    frameon : boolean
        The parameter called 'frameon' is boolean type and
        determines if frame around legend box is shown or
        not in figure.Possible inputs are 'False' and 'True'.

    loc : str or digit
        The parameter named 'loc' sets the location of the
        legend box in figure. Valid inputs are string and digits.
        For more information visit matplotlib homepage and look
        for legend section.

    numpoints : digit(optional)
        The parameter called 'numpoints' takes only digit and
        shown how many objects are shown per handlers.

    boolean_handlers : dictionary(optional)
        The parameters called 'boolean_handlers' is a dictionary
        with key names 'circle', 'lines' and 'rect' (rectangle).
        The dictionary keys takes boolean types only.

    labels_index, color_index : dictionary(optional)
        The parameters called 'labels_index' and 'color_index'
        determines both colors and labels for made legend handlers.
        Consist of the same key names as 'boolean_handlers' and
        and accepts list index type values e.g '[start:stop:step]'.

    circle_art, lines_art, rect_art : dictionary(optional)
        The parameters called 'circle_art', 'lines_art' and 'rect_art'
        are dictionaries with a different set of key names.
        Vaild key names and values are
        'x' and 'y' = digits
        'marker' = ['str'] input
        and others see code.

    Returns
    -------
    all_leg_objs, all_labels : legend objects, list
        The 'all_leg_objs' are the handlers created
        from function and 'all_labels' is a list with
        labels given.

    """

    # Local variables for legends objects and labels.
    leg_objs_circle = []
    leg_objs_lines = []
    leg_objs_rect = []
    labels_circle = []
    labels_lines = []
    labels_rect = []

    # Local dictionary variables
    handlers = {'circle': None, 'lines': None, 'rect': None}
    circle_art2 = {'x': 0, 'y': 0, 'marker': ['o'], 'linestyle': 'none',
                   'alpha': None}
    lines_art2 = {'x': 0, 'y': 0, 'linestyle': ['--'], 'marker': 'None',
                  'alpha': None, 'linewidth': None}
    rect_art2 = {'x': 0, 'y': 0, 'width': 1, 'height': 1, 'hatch': None,
                 'alpha': None}

    # 'Updating' items in dictionaries by keys
    for i in range(len(boolean_handlers)):
        handlers[boolean_handlers.keys()[i]] = boolean_handlers.values()[i]
    for i in range(len(circle_art)):
        circle_art2[circle_art.keys()[i]] = circle_art.values()[i]
    for i in range(len(lines_art)):
        lines_art2[lines_art.keys()[i]] = lines_art.values()[i]
    for i in range(len(rect_art)):
        rect_art2[rect_art.keys()[i]] = rect_art.values()[i]

    # Logic circuit for round objects in legend
    if handlers.get('circle') and len(labels_index['circle']) > 0:
        if len(labels_index['circle']) > 2:
            labels_circle = labels[labels_index['circle'][0]:
                                   labels_index['circle'][1]:
                                   labels_index['circle'][2]]
        else:
            labels_circle = labels[labels_index['circle'][0]:
                                   labels_index['circle'][1]]
        if len(color_index['circle']) > 2:
            color_circle = colors[color_index['circle'][0]:
                                  color_index['circle'][1]:
                                  color_index['circle'][2]]
        else:
            color_circle = colors[color_index['circle'][0]:
                                  color_index['circle'][1]]
        circle_objs = mpl.legend.Line2D
        leg_objs_circle = [circle_objs(
            [circle_art2.get('x')], [circle_art2.get('y')],
            marker=circle_art2.get('marker')[i],
            linestyle=circle_art2.get('linestyle'),
            alpha=circle_art2.get('alpha'),
            color=color_circle[i]) for i in range(len(labels_circle))]

    # Logic circuit line legend object
    if handlers.get('lines') and len(labels_index['lines']) > 0:
        if len(labels_index['lines']) > 2:
            labels_lines = labels[labels_index['lines'][0]:
                                  labels_index['lines'][1]:
                                  labels_index['lines'][2]]
        else:
            labels_lines = labels[labels_index['lines'][0]:
                                  labels_index['lines'][1]]
        if len(color_index['lines']) > 2:
            color_lines = colors[color_index['lines'][0]:
                                 color_index['lines'][1]:
                                 color_index['lines'][2]]
        else:
            color_lines = colors[color_index['lines'][0]:
                                 color_index['lines'][1]]
        line_objs = mpl.legend.Line2D
        leg_objs_lines = [line_objs(
            [lines_art2.get('x')], [lines_art2.get('y')],
            marker=lines_art2.get('marker'), alpha=lines_art2.get('alpha'),
            linestyle=lines_art2.get('linestyle')[i],
            linewidth=lines_art2.get('linewidth'),
            color=color_lines[i]) for i in range(len(labels_lines))]

    # Circuit for legend type rectangle
    if handlers.get('rect') and len(labels_index['rect']) > 0:
        if len(labels_index['rect']) > 2:
            labels_rect = labels[labels_index['rect'][0]:
                                 labels_index['rect'][1]:
                                 labels_index['rect'][2]]
        else:
            labels_rect = labels[labels_index['rect'][0]:
                                 labels_index['rect'][1]]
        if len(color_index['rect']) > 2:
            color_rect = colors[color_index['rect'][0]:
                                color_index['rect'][1]:
                                color_index['rect'][2]]
        else:
            color_rect = colors[color_index['rect'][0]:color_index['rect'][1]]
        rect_objs = mpl.legend.Rectangle
        leg_objs_rect = [rect_objs(
            (rect_art2.get('x'), rect_art2.get('y')),
            rect_art2.get('width'), rect_art2.get('height'), fc=color_rect[i],
            hatch=None,
            alpha=rect_art2.get('alpha')) for i in range(len(labels_rect))]

    # Sum all legend object and labels for figure.
    all_leg_objs = leg_objs_circle + leg_objs_lines + leg_objs_rect
    all_labels = labels_circle + labels_lines + labels_rect

    # Render all legends and labels to figure.
    PlotAxess.legend(
        [all_leg_objs[i] for i in range(len(all_leg_objs))],
        [r'\textbf{%s}' % (all_labels[i]) for i in range(len(all_labels))],
        numpoints=numpoints, frameon=frameon, loc=loc, fontsize=fontsize)
    return all_leg_objs, all_labels


def dataplotter_yaxis_percenatage(y, position):
    """Function used only for histogram work, where y-axis is converted
    to percentage.

    Parameters
    ----------
    y : y-axis tick values
        Tick on y-axis

    position : y-axis tick position
        Figure coordinates for y-axis tick positions

    Returns
    -------
    Transformed y-axis values.
    """
    y_values = str(100 * y)
    return y_values + r"$\%$"


def dataplotter_color_code_subframe(df, color_columns=[]):
    """The function is a helper function for x and y scatter in dataplotter
    module. The function color codes the "subframe" given in scatter plot.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The "df" is the given 'subframe' that is to be color coded for scatter
        plot.
    color_columns : list(object)
        The "color_columns" parameter are the columns that are to be colored
        coded. The values in "color_columns" must be booleans. Default value
        for "color_columns" is set to null '[]' and triggers function call
        that forces the user to pick column label(s) to use for color coding.

    Returns
    -------
    df2
        The "df2" is the given "df" object with one additional column
        called "Color_coded".

    color_columns
        The "color_columns" is list object where the items are the column(s)
        label(s) given.

    Raises
    ------
    TypeError
        If values for the labels in "color_columns" is not boolean typed.
    ValueError
        If number of labels in "color_columns" exceeds 25.

    See Also
    --------
    phen_o_links.dataset_pick_columns : For more information about function
                                        called made when "color_columns" is set
                                        to null.

    """
    #Local global
    palette = ['green', 'red', 'blue', 'yellow', 'colorblind']
    hues_2 = ['light', 'lightdark', 'darklight', 'dark']
    type_c = np.array([False])

    # Coping frame
    df2 = ds.dataset_copy_frame(df)

    # Picking columns
    if not color_columns:
        color_columns, idx = ds.dataset_pick_columns(df2, split='groupby')
        del idx

    if not df2[color_columns].values.dtype == type_c.dtype:
        text = ("The 'color_columns' given have items that are not "
                "boolean\n:{0}").format(df2[color_columns].values[0])
        raise TypeError(text)

    #Ordering input!
    color_columns = df2[color_columns].sum()
    color_columns = color_columns.sort_values(ascending=False)
    color_columns = color_columns.index.tolist()

    # Creating order of groups
    df2['Nr_True'] = [
        np.max(np.nonzero(i != 0)[0]) for i in df2[color_columns].values]
    df2 = df2.sort_values(by='Nr_True')

    # Removing overlapping categories
    columns_dict = {i:v for i,v in enumerate(color_columns)}
    color_columns = [columns_dict.get(i) for i in df2.Nr_True.unique()]

    # Creating color palette
    if len(color_columns) <= 5:
        palette = palette[:len(color_columns)]
        hues_2 = []

    if len(color_columns) > 5:
        remain = len(color_columns) % 5
        add_to = 5 - remain
        n_slice = int((len(color_columns) + add_to) / 5)
        hues_2 = hues_2[::-1]
        hues_2 = hues_2[:n_slice]

    if len(color_columns) > 25:
        text = ("Please reconsider the amounts of groups to color code, n=25"
                "or more. This will not work well!")
        raise ValueError(text)
    # Making colors coding
    palette_c, d_c, dl_c, ld_c, l_c = dataplotter_colorscheme(main=palette,
                                                              hues=hues_2)
    color_theme = palette_c + d_c + dl_c + ld_c + l_c
    color_dict = {
        df2.Nr_True.unique()[i]:color_theme[i] for i in range(
            len(df2.Nr_True.unique()))}

    df2 = ds.dataset_add_column_by_dict(df2, color_dict, Grouper="Nr_True",
                                        new_col="Color_coded")
    #del df2['Nr_True']

    # Ordering labels to match color_coding
    t_table = df2[color_columns].sum()
    t_table = t_table.sort_values(ascending=False)
    color_columns = t_table.index.tolist()
    color_columns = [
        i.replace('_', ' ') for i in color_columns]
    color_columns = dataplotter_textspacemanger(color_columns)
    df2['Freq'] = df2.groupby('Nr_True')['Nr_True'].transform('count')
    df2 = df2.sort_values(by='Freq', ascending=False)

    return df2, color_columns


# Main worker functions


def dataplotter_barplot(
        df, run_batch=False, batch=[],
        color=['green', 'red'], batch_color=['yellow', 'blue'],
        labels=[], labels2=[], figure_text=['Untitle', 'Yaxis', 'Xaxis']):

    """The function returns a bar plot. The minimum requirements to run
    the function are the values are ordered column wise. The parameters
    called batch are specif and are only used if user flags for it. Batch
    rendering is only valid for 2 columns and 2 batch values, all other
    combination result in index error!


    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called "df" is the data input.

    run_batch : boolean (optional)
        The parameter called "run_batch", accepts booleans values and
        determines if batch effect is incorporated in bar plot.

    batch : list(object)
        The parameter called "batch" accepts only one
        pandas.core.frame.DataFrame(object).
    color, batch_color : list(object)
        The parameters called "color" and "batch_color" are list objects.
        Valid entry are string entries found in dataplotter_colorscheme.

    labels, lables2 : list(object)
        The parameters called "labels" and "labels2" are the text for legend
        box in figure.
    figure_text : list(optional)
        The parameter called "figure_text" has three entries all strings. Order
        of entry is figure title, y-axis and lastly x-axis.

    Returns
    -------
    fig1 : matplotlib.figure.Figure(object)
        The "fig1" is the stored bar plot.

    ax1 : matplotlib.axes.AxesSubplot(object)
        The "ax1" is the axis of rendered figure
    """

    # Fixing text
    dataplotter_fixing_textformat()
    # Local global
    data_columns = []
    batch_index = []
    batch_columns = []
    bar_data = []
    bar_error = []
    bar_width = 0.35
    batch_data = []
    batch_error = []
    data_data = []
    data_error = []
    data_index = []

    # Copying main data
    df_work1 = ds.dataset_copy_frame(df)

    # Check that parameters are empty for data
    if not any(data_columns) or not any(data_index):
        text = "Here you pick the data that are the bars! "
        print text
        data_columns, data_index2 = ds.dataset_pick_columns(df_work1,
                                                            split='groupby')
        # Getting data
        for i in data_columns:
            data_data.append(df_work1[i].values)
        data_index.append(df_work1[data_index2].values)

    # Check for option2
    option2 = raw_input("Do you wish to add error bars to data Y/N:\t")

    option2 = option2.lower()

    if option2 == 'y':
        text2 = "Here you pick columns with error margins e.g. std "
        print text2
        data_error2, data_idx_error = ds.dataset_pick_columns(df_work1,
                                                              split='groupby')
        del data_idx_error
        # Getting data
        for i in data_error2:
            data_error.append(df_work1[i].values)
    if option2 == 'n':
        text3 = "No error bars are plotted in figure."
        print text3

    # Check if value run batch parameter

    if not run_batch:
        text4 = "No batch effects bars are plotted"
        print text4

    if run_batch:
        text5 = " Batch effects bars are plotted"
        print text5

        # Ordering
        batch = batch[0]

        # Copying
        batch = ds.dataset_copy_frame(batch)

        if not any(batch_columns) or not any(batch_index):
            text6 = ("First round ordering batch table" +
                     " by index given in column order and the column(s)" +
                     " with wanted values!")
            print text6

            batch_columns, batch_index = ds.dataset_pick_columns(batch,
                                                                 split='groupby')

            batch2 = batch.set_index(batch_index[0])

            batch2 = batch2.T

            text7 = "Now pick value for batch bar(s)"
            print text7

            # Picking values
            batch2_values, batch_index2 = ds.dataset_pick_columns(batch2,
                                                                  split='groupby')
            del batch_index2

            batch_data.append(
                batch2[batch2_values[0]][[i for i in batch_columns]].values)

            text8 = "Now pick value for batch error bar(s)"
            print text8

            # Picking values
            batch3_values, batch_index3 = ds.dataset_pick_columns(batch2,
                                                                  split='groupby')

            del batch_index3

            batch_error.append(
                batch2[batch3_values[0]][[i for i in batch_columns]].values)

    # Ordering data
    data_index = list(np.ravel(data_index))
    batch_xlabel = []

    # Fixing data
    for i in range(len(data_data)):
        tmp = list(data_data[i])
        bar_data.append(tmp)

    # Conditionals execution
    if data_error:
        for i in range(len(data_error)):
            tmp = list(data_error[i])
            bar_error.append(tmp)
    if batch_data:
        batch_xlabel = ['batch']
        for i in range(len(batch_data[0])):
            bar_data[i].append(batch_data[0][i])
    if batch_error:
        for i in range(len(batch_data[0])):
            bar_error[i].append(batch_data[0][i])

    # Formatting x_ticks
    bar_xtickslabels = data_index + batch_xlabel
    bar_xtickslabels = [str(i) for i in bar_xtickslabels]

    # Creating new index and bars width!
    new_index = np.arange(len(bar_error[0]))
    bar_width = 1.0/len(data_columns) * 0.7

    # Creating color scheme
    # For main data
    m, l, ld, dl, d = dataplotter_colorscheme(
        main=color, hues=['light', 'lightdark', 'darklight', 'dark'])
    # For batch
    pick_color2 = []
    if run_batch:
        m2, l2, ld2, dl2, d2 = dataplotter_colorscheme(
            main=batch_color, hues=['light', 'lightdark', 'darklight', 'dark'])
        colorset2 = np.asanyarray((m2, l2, ld2, dl2, d2))
        color2_ix = np.hsplit(colorset2, len(batch_color))
        for i in color2_ix:
            n2 = (np.random.choice(i.ravel(), 1))
            pick_color2.append(n2.tostring())

    # For main set of data
    colorset = np.asanyarray((m, l, ld, dl, d))
    color_ix = np.hsplit(colorset, len(color))
    pick_color = []
    for i in color_ix:
        n = (np.random.choice(i.ravel(), 1))
        pick_color.append(n.tostring())
    # Creating canvas
    fig1 = plt.figure(figsize=(8, 6))

    # Creating figure
    ax1 = fig1.add_subplot(111)

    # Cleaning axis
    dataplotter_x_y_tick_remover(ax1)
    dataplotter_spines_remover(
            ax1, top=False, bottom=True, left=True, right=False)
   # return bar_data

    if len(data_index) == len(new_index):
        for i in range(len(bar_data)):
            calc = 0
            if i > 0:
                calc = i * bar_width
            ax1.bar(new_index + calc, bar_data[i], bar_width,
                    yerr=bar_error[i], label=labels[i], color=pick_color[i],
                    ecolor='black')
    if len(data_index) != len(new_index):
        for i in range(len(bar_data)):
            calc = 0
            if i > 0:
                calc = i * bar_width
            ax1.bar(new_index[:len(data_index)] + calc,
                    bar_data[i][:len(data_index)], bar_width,
                    yerr=bar_error[i][:len(data_index)], label=labels[i],
                    color=pick_color[i],ecolor='black')
            ax1.bar(new_index[len(data_index):] + calc,
                    bar_data[i][len(data_index):], bar_width,
                    yerr=bar_error[i][len(data_index):], label=labels2[i],
                    color=pick_color2[i], ecolor='black')

    plt.xticks(new_index + bar_width, bar_xtickslabels)
    plt.legend(bbox_to_anchor=(1.05,1),loc=2, borderaxespad=0.)
    plt.title(figure_text[0])
    plt.ylabel(figure_text[1])
    plt.xlabel(figure_text[2])
    plt.show()

    return fig1, ax1


def dataplotter_bar_plot_simple(
    df, columns=[], index=[], figlabels=["Title", "X axis", "Y axis"],
    datalabels=[],y_log=False, rot=90.0):
    """ Take a given pandas data frame and returns a simple bar plot.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The 'df' is a pandas data frame with the data.

    columns, index : list(optional)
        The parameters called 'columns' and 'index' are the column labels
        present in 'df'. The 'columns' is the individual bars plotted
        and the 'index' is the x-axis labels under each bar. If parameters
        are left empty, function call is triggered.

    figlabels : list(optional)
        The 'figlabels' is list object with the length of 3, which renders the
        figure text. The order of nth- items in 'figlabels' corresponds to
        different properties of text. The first item is figure title, seconds
        item is the x-axis label and the last item is the y-axis label.

    datalabels : list(optinal)
        The 'datalabels' is an optional list. The items in "datalables" is the
        legend text of the figure. The items corresponds to the order columns.

    rot : float(optional)
        The 'rot' rotates the x-axis major tick labels a certain degrees.

    Returns
    -------
    fig1 : matplotlib.figure.Figure(object)
        The 'fig1' is the figure plotted.

    ax1 : matplotlib.axes._subplots.AxesSubplot(object)
        The 'ax1' is the axes of the figure object.

    Raises
    ------
        None error have been accounted for!

    See Also
    --------
    dataplotter_save_figure : "fig1" return is used for save.

    dataplotter_x_y_tick_remover : For information about tick remover

    dataplotter_fixing_textformat : For more information about text rendering

    phen_o_links.dataset.dataset_copy_frame : For more information about
                                              data frame copy.

    phen_o_links.dataset.dataset_pick_columns : For more information about
                                                function call if 'columns'
                                                or 'index' is left empty.
    """

    # Copying main data
    rot = float(rot)
    df1 = ds.dataset_copy_frame(df)

    # Fixes removes under score from column label(s).
    df1.columns = [i.replace("_", " ") for i in df1.columns.tolist()]

    # Check that inputs are correct
    if not(columns and index):
        columns, index = ds.dataset_pick_columns(df1, split="groupby")

    # Converting to latex styled text
    dataplotter_fixing_textformat()
    figure_text = dataplotter_textspacemanger(figlabels)

    # Making y-ticks
    max_tick = np.ceil(np.log10(df1.max().max()))
    y_ticks = np.arange(0, max_tick*10**(max_tick-1)+1,100)[::2]

    # Creating figure
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(1,1,1)

    # Plotting things
    df1[columns].plot(kind="bar", ax=ax1)

    # Removing y and x tick
    dataplotter_x_y_tick_remover(ax1)


    # Plotting bars
    ax1.xaxis.set_ticklabels(
        list(np.ravel(df1[index].values)),rotation=rot)
    ax1.set_yticks(list(y_ticks))

    if y_log:
        ax1.set_yscale("log")
        ax1.yaxis.set_tick_params(which="minor", left="off")
        y_ticks = np.ceil(np.logspace(0,max_tick,len(y_ticks)))
        y_ticks = set(y_ticks)
        ax1.yaxis.set_ticks(list(y_ticks))

    # Checking for "datalabels" option
    h, l = ax1.get_legend_handles_labels()

    if datalabels:
        datalabels= dataplotter_textspacemanger(
            datalabels, pattern="_", output=" ")
        datalabels = dataplotter_textspacemanger(datalabels)
        if not(len(l) == len(datalabels)):
            text=("Legend labels are not changed!")
            print text
            datalabels = l
        l = datalabels

    # Adding text to figure
    ax1.legend(h,l, frameon=False)

    plt.title(r"%s" % (figure_text[0]))
    plt.ylabel(r"%s" % (figure_text[2]))
    plt.xlabel(r"%s" % (figure_text[1]))

    plt.show()
    print "Don't for get to save figure!"

    return fig1, ax1


def dataplotter_scatter_x_y_plot(
        df, filter_val, sub_limit, percentage, number=5,
        extra_features=[False, False, False, False, True, False],
        func_call=[False, False, False], sub_frame=[],
        markersize=20, fig_title='Untitled',
        x_title='Untitled', y_title='Untitled', datapoints='Untitled',
        regtext='Untitled', fig_fontsize=[12, 10, 8], all_axis=False,
        spines=[False, True, False, True], a_txt=True, c_txt=True, trn=0.5,
        trn2=0.5):
    """Takes a data frame object from pandas and returns scatter plot
    of two columns either with or without regression line.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        Input data frame.

    filter_val : float, int
        Value for filtering picked columns. The "filter_val" parameter also
        set the x-axis and y-axis upper limits of figure.

    sub_limit : float, int
        The "sub_limit" parameters is used to set sub limit for figure.
        Recommend to be of negative value.

    percentage : float
        The 'percentage' accepts floats values between 0.0-1.0
        recommend value is to set percentage to 0.05.

    number : int (optional)
        The 'number' parameter returns number of assigned outliers
        in a dataset. The 'number' values is 1/2 of the outliers
        meaning that Outliers = number * 2.

    extra_features : list (optional)
        The 'extra_features' is list that only accepts boolean values.
        The parameter adds extra features upon rendered scatter figure.
        The order in 'extra_features' have different properties
        'extra_features' = ['Identity Line', 'Grid Line','Linear Regression',
        'Frame around figure legend','Shadow behind figure legend', 'Special
        Grid Line']. The 'Special Grid Line' renders a regression line with the
        values that are colored gray in Figure. The default setting is all
        'extra_features' options to False except for 'Shadow behind figure
        legend'.

    func_call : list (optional)
        The 'func_call' accepts only boolean values. The item order calls
        different functions, where 'outliers' are assigned. The 'func_call'
        default setting is 'func_call' = [False, False, False], and no
        functions are executed.

    sub_frame : list(optional)
        The 'sub_frame' parameter has 2 valid type inputs in the following
        order 'sub_frame' = [df(object), str(object)]
        If str(object) is set to "Colorcode" and function for coloring
        subframe is activated.If sub_frame option is in use, the 'func_call'
        is disable.

    markersize : float, int (optional)
        The parameter called 'markersize' accepts float or int values
        and determines the size for all scatter marker in use.
        The default setting for 'markersize'= 20.

    fig_title : str (optional)
        Title for figure default value for "fig_title" is 'Untitled'.

    x_title : str (optional)
        X-axis title in figure. Default setting for "x_title" is 'Untitled'.

    y_title : str (optional)
        Y-axis title in figure. Default setting for "y_title" is 'Untitled'.

    datapoints : str (optional)
        Legend title for data points. Default setting
        for "datapoints" is 'Untitled'.

    figure_font_size : list (optional)
        The parameter called 'figure_font_size' accepts floats or integers
        as item input. The items-nth order is coupled to a specific
        features see below
        figure_font_size = [Title, Axis-label, Text in figure]
        The default values are 12, 10 and 8 in font size.

    all_axis : boolean(optional)
        The parameters removes spines from figure. The default setting
        for "all_axis" is False. If "all_axis" is True all spines are removed
        from figure.

    spines : [Top, Bottom, Right, Left] boolean (optional)
        The parameters takes 4 boolean values in total.
        The input order is Top, Bottom, Right and Left for figure spines.
        The values determines visibility of spines a True equals to visible.
        The default values for "spines" is [False, True, False, True].

    a_txt, c_txt : boolean(optional)
        The parameter called 'a_txt' accepts only booleans and is responsible
        for visualizes point annotation given by 'index' set by functional.
        Default setting for 'a_txt' is set to 'True'

    trn, trn2 : float(optional)
        The 'trn' parameter set the transparency of the points from 0.0 - 1.0.
        The 'trn' is set as default to '0.5'. The 'trn2' specifies significant
        dots.

    Returns
    -------
    fig1
        The "fig1" is a plt.figure (object). The figure displayed with sub
        plot. For more information about figure objects search in Matplotlib
        documentation matplotlib.figure.Figure

    ax
        The added axes for the "fig1"(object). More information about subplot
        axes search for matplot.axes._subplots.AxesSubplot in Matplotlib
        documentation.

    Raises
    ------
    ValueError
        If "sub_frame" labels are not found in main data frame ('df').

    See Also
    --------
    dataplotter_save_figure : "fig1" return is used for save.

    dataplotter_spines_remover : For more information about "all_axis" and
                                 "spines".
    dataplotter_x_y_tick_remover : For information about tick remover

    dataplotter_fixing_textformat : For more information about text rendering
                                    in figure.

    dataplotter_x_y_limit : For more information about axis
                            limitation with "filter_val".

    dataplotter_color_code_subframe : For more information about color coding
                                      "sub_frame".


    phen_o_links.dataset.dataset_copy_frame : For more information about
                                              data frame copy.

    phen_o_links.dataset.dataset_top_and_bottom : For more information about
                                                  'num', 'percentage' and
                                                  'func_call'.

    phen_o_links.dataset.dataset_filter_by_value : For more information about
                                                   "filter_val" parameter.

    phen_o_links.dataset.dataset_regline : For more information about
                                           regression line.

    Examples
    --------
    >>> # Applying Colorcoded keyword in sub_frame parameter
    >>> import pandas as pd
    >>> import phen_o_links.dataplotter as dp

    >>> # Importing pandas data frame
    >>> df = pd.read_csv('file.csv', delimiter='\\t')

    >>> # Columns present in df
    >>> df.columns
    Index([u'Unique_Label', u'X_values', u'Y_values', u'Test', u'Test2',
          u'Test3'], dtype='object')

    >>> df.shape
    (150, 6)

    >>> # Top five values
    >>> df.head(5)
    Unique_Label    X_values    Y_values    Test    Test2   Test3
               A           1          10    True    False   False
               B           2          11    True    False   False
               C           4          12    True    True    False
               D           8          13    True    True    False
               E          16          14    True    True    True

    >>> # Creating sub frame called sub
    sub = df[df.Test==True]

    >>> # Creating scatter plott
    fig1, ax1 = dp.dataplotter_scatter_x_y_plot(
        df,20,-20, 0.05 ,sub_frame=[sub, 'Colorcoded'])

    >>> # Columns in df prompt
    {0:'Unique_Label'
     1:'X_values'
     2:'Y_values'
     3:'Test'
     4:'Test2'
     5:'Test3'}

    >>> # User picked labels
    Pick columns to work with via the numbers e.g. 1,2,3 etc
        : 1,2,
    Pick a indexer or groupby for data frame here
        :0,

    >>> # Picking columns to colorcoded sub frame values
    Entering 'colorcode' mode!
    Pick columns to work with via the numbers e.g. 1,2,3 etc
        : 3,4,5,
    Pick a indexer or groupby for data frame here
        :0,

    >>> # Saving figure as svg and pdf formats
    dp.dataplotter_save_figure(fig=fig1, filename='test',
                               save_as=['svg', 'pdf'])
    """
    # Fixing text
    dataplotter_fixing_textformat()

    # Local
    df_grey =[]

    # Getting colors
    m1, l1, ld1, dl1, d1 = dataplotter_colorscheme(
        main=['colorblind'], hues=['lightdark', 'darklight'])
    m, l, ld, dl, d = dataplotter_colorscheme(
        main=['grayscale2'], hues=['lightdark', 'darklight'])
    m2, l2, ld2, dl2, d2 = dataplotter_colorscheme(
        main=['blue'], hues=['lightdark', 'darklight'])
    # Handling returns
    gray = m + l + ld + dl + d
    gray = gray[1:]
    blue = m2 + l2 + ld2 + dl2 + d2

    # Picking colors
    color = list(np.random.choice(blue, 1))
    color2 = list(np.random.choice(gray, 1))

    # Checking that sub_frame is on
    if sub_frame:
        func_call = [False, False, False]

    # Copying data frame
    df_work1 = ds.dataset_copy_frame(df)

    # Creating and subset and filtering data frame
    df_w, df_work2, x, y, columns = ds.dataset_filter_within_a_range_of_value(
        df_work1, filter_val)
    # Deleting unwanted return.
    del df_w

    # Dividing columns str into 2 groups.
    indexer = columns[:-2]
    columns = columns[-2:]

    # Getting regression line
    slope, intercept, r_value, p_value, std_err = ds.dataset_regline(x, y)

    x = np.hstack(((filter_val,sub_limit),x))

    # Setting x and y for identity line plot
    x1 = np.linspace(sub_limit, filter_val, num=3)
    y1 = np.linspace(sub_limit, filter_val, num=3)

    # Setting canvas and subplots for the plot
    fig1 = plt.figure(figsize=(8, 6))

    # Creating axes
    ax = fig1.add_subplot(111, aspect='equal')

    # Used for plotting regression line
    line = slope * x + intercept

    # Creating nicer line for regression
    x_reg = x
    y_reg = line

    # Text formatting
    text_fig = [fig_title, x_title, y_title, datapoints, regtext]
    text_latex = dataplotter_textspacemanger(text_fig)

    # Setting Title to figure
    ax.set_title(r'\textbf{%s}' % (text_latex[0]), fontsize=fig_fontsize[0])
    ax.set_xlabel(r'\textbf{%s}' % (text_latex[1]), fontsize=fig_fontsize[1])
    ax.set_ylabel(r'\textbf{%s}' % (text_latex[2]), fontsize=fig_fontsize[1])

    # Setting axis limit
    dataplotter_x_y_limit(ax, x_limit=filter_val,
                          y_limit=filter_val, x_mini=sub_limit,
                          y_mini=sub_limit)

    # Removing ticks and spines!
    dataplotter_x_y_tick_remover(ax)
    dataplotter_spines_remover(
        ax, all_axis=all_axis, top=spines[0], bottom=spines[1],
        right=spines[2], left=spines[3])

    if sum(func_call) == 0 and not sub_frame:
        # Plotting raw data points
        ax.scatter(x, y, c=blue[0], marker='o', s=markersize,
                   alpha=trn, label=r"{%s}" % (text_latex[3]), lw=0)

    if sum(func_call) > 0:
        # Function call.
        subset, idx2, columns2, func_names = ds.dataset_top_and_bottom_extremes(
            df_work2, number, percentage, columns=columns,
            indexer=[indexer[0]], func_call=func_call)

        # Fixing function call
        func_names = dataplotter_textspacemanger(func_names)

        # Check length func names
        if len(func_names) % 2 and len(func_names) > 1:
            func_names = [
                func_names[0] + '\n' + func_names[1] + '\n' + func_names[2]]

        if len(func_names) == 2:
            func_names = [func_names[0] + '\n' + func_names[1]]

        # Slicing away 'outliers' from main pandas frame.
        df_grey = df_work2[~(df_work2[indexer[0]].isin(subset[indexer[0]]))]

        # Slicing for 'outliers' from main pandas frame.
        df_outlier = df_work2[(df_work2[indexer[0]].isin(subset[indexer[0]]))]

        # Plotting insignificant points in scatter figure.
        ax.scatter(df_grey[columns[0]].values,
                   df_grey[columns[1]].values, c=color2, marker='o',
                   s=markersize, lw=0,
                   alpha=trn, label=r"{%s}" % (text_latex[3]))

        # Plotting significant points
        ax.scatter(df_outlier[columns[0]].values,
                   df_outlier[columns[1]].values, c=color, marker='o',
                   s=markersize, lw=0, alpha=trn2,
                   label=r"{%s}" % (func_names[0]))

        if a_txt:
            # Points are getting names.
            dataplotter_text_annotation_scatter(
                ax, df_outlier, picked_columns=columns,
                indexer=[indexer[-1]], fontsize=fig_fontsize[2],
                color=color[0])

    if sub_frame:
        # Sub parting sub frame input into different variables
        text = [sub_frame[-1]]
        subset2 = sub_frame[0]

        if not text[0] == 'Colorcoded':
            # Fixing 'blank space' problems for latex rendering.
            text_latex2 = dataplotter_textspacemanger(text)

            # Creating pandas.core.frame.DataFrame(object)
            check_columns = [indexer[0]] + columns

            # List present columns in subset
            subset_columns = subset2.columns.tolist()
            subset_columns2 = [i for i in subset_columns if i in check_columns]

            # Check that if indexer is present in sub frame contains indexer
            try:
                [subset_columns2.index(i) for i in check_columns]

            except ValueError:

                prompt = ("\n User 'sub_frame' does not contain all"
                          " valid columns."
                          "\nValid option columns options are :\n {0}\n "
                          "\nUser 'sub_frame'"
                          " columns were"
                          ": {1}\n").format(check_columns, subset_columns2)
                print prompt
                return

        # Slicing away 'outliers' from main pandas frame.
        df_grey = df_work2[~(df_work2[indexer[0]].isin(subset2[indexer[0]]))]

        # Slicing for 'outliers' from main pandas frame.
        df_outlier2 = df_work2[
            (df_work2[indexer[0]].isin(subset2[indexer[0]]))]

        # Plotting insignificant points in scatter figure.
        ax.scatter(df_grey[columns[0]].values,
                   df_grey[columns[1]].values, c=color2, marker='o',
                   s=markersize, lw=0,
                   alpha=trn, label=r"{%s}" % (text_latex[3]))

        if not text[0] == 'Colorcoded':
            # Plotting significant points
            ax.scatter(df_outlier2[columns[0]].values,
                       df_outlier2[columns[1]].values, c=color, marker='o',
                       s=markersize, lw=0, alpha=trn2,
                       label=r"{%s}" % (text_latex2[0]))

        if text[0] == 'Colorcoded':
            print "Entering 'colorcoded' mode!"
            df_outlier3, label_colors = dataplotter_color_code_subframe(
                subset2)
            m_color = df_outlier3.Color_coded.unique()
            #m_color = list(m_color)
            gr = df_outlier3.groupby('Color_coded')
            for i in range(len(m_color)):
                ax.scatter(
                    gr.get_group(m_color[i])[columns[0]].values,
                    gr.get_group(m_color[i])[columns[1]].values,
                    c=m_color[i], marker='o', s=markersize, lw=0,
                    alpha=trn2, label=r"{%s}" % (label_colors[i]))

        if c_txt:
            print "Color coded Annotations for groups size of n <= 30!"
            t_frame = pd.DataFrame(gr.size() <= 30, columns=['Trues'])
            t_db = t_frame.sort_values('Trues', ascending=False)
            n_db = np.nonzero(t_db.Trues)[0][-1] + 1
            t_db = t_db[:n_db]
            left_c = t_db.index.tolist()
            df_outlier3['Outlier_c'] = df_outlier3.Color_coded.isin(left_c)
            df_outlier2 = df_outlier3[df_outlier3.Outlier_c == True]

        if a_txt:
        # Points are getting names.
            dataplotter_text_annotation_scatter(
                ax, df_outlier2, picked_columns=columns,
                indexer=[indexer[-1]], fontsize=fig_fontsize[2],
                color=color[0])

    # Optional parts

    if extra_features[0]:
        # Plotting identity line
        ax.plot(x1, y1, 'k', linewidth=2, alpha=0.5,
                label=r"\textbf{Identity\ Line}")

    if extra_features[1]:
        # Sets horizontal line
        ax.axhline(0, color='k')
        # Sets vertical line
        ax.axvline(0, color='k')

    if extra_features[2]:

        # Plotting regression line
        ax.plot(x_reg, y_reg, '--k', linewidth=2, alpha=0.375,
                label=r"\textbf{%s}" % (text_latex[4]))

        # Text formatting for equation and correlations
        ax.text(0.5, 0.975,
                r"\textit{%s}" % (text_latex[4] + r'$\ r^{2}$' + " = ")
                + r"$%3.7s$" % (r_value ** 2),
                fontsize=fig_fontsize[2], transform=ax.transAxes)
        ax.text(0.5, 0.95,
                r"\textit{%s}" % (text_latex[4] + " equation\  = ")
                + r"$%3.7s x + %3.7s$" % (slope, intercept),
                fontsize=fig_fontsize[2], transform=ax.transAxes)

        ax.text(0.5, 0.925,
                r"\textit{%s}" % (text_latex[4] + r"$\ \sigma_{est}$" + " = ")
                + r"$%3.7s$" % (std_err),
                fontsize=fig_fontsize[2], transform=ax.transAxes)

    if extra_features[5]:
        text_latex[4] = r"Regression\ Line\ Special"
        # Getting values
        x, y = df_grey[columns[0]].values, df_grey[columns[1]].values
        # Getting regression line
        slope, intercept, r_value, p_value, std_err = ds.dataset_regline(x, y)
        x = np.hstack(((filter_val,sub_limit),x))

        #Linear regression line
        y_reg = slope * x + intercept

        # Plotting regression line
        ax.plot(x, y_reg, '-.k', linewidth=2, alpha=0.375,
                label=r"\textbf{%s}" % (text_latex[4]))

        # Text formatting for equation and correlations
        ax.text(0.5, 0.90,
                r"\textit{%s}" % (text_latex[4] + r'$\ r^{2}$' + " = ")
                + r"$%3.7s$" % (r_value ** 2),
                fontsize=fig_fontsize[2], transform=ax.transAxes)
        ax.text(0.5, 0.875,
                r"\textit{%s}" % (text_latex[4] + " equation\  = ")
                + r"$%3.7s x + %3.7s$" % (slope, intercept),
                fontsize=fig_fontsize[2], transform=ax.transAxes)

        ax.text(0.5, 0.85,
                r"\textit{%s}" % (text_latex[4] + r"$\ \sigma_{est}$" + " = ")
                + r"$%3.7s$" % (std_err),
                fontsize=fig_fontsize[2], transform=ax.transAxes)
    # Legend formatting
    ax.legend(loc='upper left', shadow=extra_features[4],
              frameon=extra_features[3], fontsize=fig_fontsize[2])
    plt.show()

    # Returns fig and axes
    return fig1, ax


def dataplotter_boxplot(
        df, zoom_procent=[90, 90], zoom=False, fig_title=['Untitled'],
        x_title=['Untitled'], y_title=['Untitled'], box_label=['Untitled'],
        boxdata=['Untitled'], fontsize=[12, 10, 8], group=[],
        n=1, loc=(1.05, 0.8), frameon=False, split='groupby'):
    """The returns a box plot of input data frame.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        Input data frame

    zoom_procent : list
        The 'zoom_procent' parameters is set to 90 percent
        for y-axis positive and y-axis negative values. The zoom action
        has an upper limit of 100 and mini of 0. The parameters takes pairs
        floats or integers.

    zoom : boolean
        The 'zoom' parameter is set to 'False' as default setting.
        If change to 'True' a zoomed version of the figure will
        rendered.

    fig_title, x_title, y_title, box_label, boxdata : list
        These parameter are used for text annotation
        in rendered box figure. All inputs are for list items
        is string and default settings is 'Untitled'.
        The parameter called 'boxdata' is used for
        custom legend rendering. The 'box_label' are used
        for labelling minor thick per box.

    fontsize : list
        The 'fontsize' parameter is a list with valid
        input items are digits. The index order determines
        the size in points for ['figure title',
        'x and y labels' and 'text in other figures']. The default
        settings are 12, 10 and 8.

    group : list
        The 'group' is list object that takes 1 item and integer and
        uses that account for how users determines data grouping.

    n : int
        The parameter called 'n' is set to '1' and decides the amount of
        points shown in legend.

    loc : int, str, tuple
        The parameters called 'loc' is set to tuple(1.05,0.80) and
        decides the placement
        for legend box in figure.

    frameon : boolean
        The 'frameon' is set to 'False' and renders
        frame around legend box if set to 'True'

    split: str(optional)
        The split parameter is used to split data.

    Returns
    -------
    fig1, ax1 : matplotlib.Figure(object), matplotlib.axes(object)
        The 'fig1' and 'ax1' are figure and respective axes
        for rendered figure.

    fig1, fig2 , ax1 , ax2 : matplotlib.Figure(object), matplotlib.axes(object)
        The 'fig1' is original figure without
        zoom and 'fig2' is zoomed version. The 'ax1' and 'ax2' follow
        the same logic as 'fig1' and respective 'fig2'.

    Raises
    ------
    ZeroDivisionError
        If number of boxes is unequal to given legend names.

    See Also
    --------
    phen_o_links.dataset.dataset_pick_columns : For more information about
                                              "split"

    """

    # Global Local
    boxplot_data=[]
    sort_by = []

    # Getting Latex rendering
    dataplotter_fixing_textformat()

    # Formatting text entery
    text_str = fig_title + x_title + y_title
    boxdata = ['Outliers', 'Median', 'Whiskers'] + boxdata

    # Formats spacing for latex
    latex_str = dataplotter_textspacemanger(text_str)
    latex_str2 = dataplotter_textspacemanger(box_label)
    latex_str3 = dataplotter_textspacemanger(boxdata)


    # Copy input data frame
    df_work1 = ds.dataset_copy_frame(df)

    # Picking columns.
    work, index = ds.dataset_pick_columns(df_work1, split=split)

    if not group:
        group = len(work)

    if group:
        if not(all(isinstance(i, int) for i in group) and len(group) == 1):
            txt = ("The group parameter accepts only 1 item and only integers."
                   "\n User input:\n {0}").format(group)
            raise ValueError(txt)
        group = group[0]

    answer = str(raw_input('Do you wish to split data frame'
                           '\n by given index: press either Y/N \n\t'))

    # Logic circuit for index split data
    if answer.lower() == 'y':
        df_gr = df_work1.groupby([i for i in index])
        for i in (df_gr.indices.keys()):
            sort_by.append(i)
        sort_by.sort()
        print sort_by
        for i in sort_by:
            for names in work:
                boxplot_data.append(
                    df_gr.get_group(i)[names].values)
        print(len(boxplot_data))

    elif answer.lower() == 'n':
        # Creating variable for plotting.
        boxplot_data = [df_work1[work[i]].values for i in range(len(work))]
    else:
        print "User must state either 'No' or 'Yes' to continue!"
        return

    # Color scheme
    colorscheme, light, light_dark, dark_light, dark = dataplotter_colorscheme(
        main=['green', 'red', 'blue',
              'yellow', 'black'], hues=['light', 'lightdark',
                                        'darklight', 'dark'])
    colorscheme2 = colorscheme + light + light_dark + dark_light + dark

    # Picking colors for legend items.
    flier_c = [light[1]] + [dark_light[1]] + [dark_light[-1]]

    # Getting rid of colors for legend items
    [colorscheme2.pop(colorscheme2.index(i)) for i in flier_c]

    # Getting rid of None values
    for i in range(colorscheme2.count(None)):
        colorscheme2.pop(colorscheme2.index(None))

    # Getting unique colors schemes!
    picked_color = []
    while len(picked_color) < group:
        picked_color = np.random.choice(colorscheme2, group)
        unique_colors = np.unique(picked_color)
        picked_color = unique_colors
    picked_color = picked_color.tolist()

    # Adding colors to picked_color list!
    if answer.lower() == 'y':
        # Finding factor to multiply
        if not len(sort_by) % group:
            factor = (len(sort_by) / group) * group
            picked_color = picked_color * factor
            print 'New more colors', len(picked_color)
        else:
            print ("Please check that the 'group'"
                   " is divisible to whole number"
                   " for:\n\t {0}").format(len(sort_by))
            return
    all_colors = flier_c + picked_color


    # Trying out if box color(s) picked are equal to box data.
    #try:
    #    1.0 / (len(picked_color) == (len(boxdata) - 3))

    #except ZeroDivisionError:
    #    print 'Legend labels and amount of boxes differ'
    #    print 'Number of boxes: {0} and number of legends {1}'.format(
    #        len(picked_color), (len(boxdata) - 3))
    #    return

    # Creating Figures and axes.
    fig1 = plt.figure(figsize=(8, 6))
    fig2 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    # Adding Subplots type box plot.
    boxplot1 = ax1.boxplot(boxplot_data, patch_artist=True)
    boxplot2 = ax2.boxplot(boxplot_data, patch_artist=True)

    # Creating labels for for boxes in figure.
    ax1.set_xticklabels([r'{%s}' % (i) for i in latex_str2],
                        fontsize=fontsize[2])

    ax1.set_title(r'\textbf{%s}' % (latex_str[0]), fontsize=fontsize[0])
    ax1.set_xlabel(r'\textbf{%s}' % (latex_str[1]), fontsize=fontsize[1])
    ax1.set_ylabel(r'\textbf{%s}' % (latex_str[2]), fontsize=fontsize[1])

    # Creating objects for legend.
    org_leg_objs, org_labels = dataplotter_customlegends(
        ax1, labels=latex_str3, colors=all_colors,
        boolean_handlers={'circle': True, 'lines': True, 'rect': True},
        fontsize=fontsize[2], loc=loc, numpoints=n, frameon=frameon,
        labels_index={'circle': [0, 1], 'lines': [1, 3],
                      'rect': [3, len(latex_str3)]},
        color_index={'circle': [0, 1], 'lines': [1, 3],
                     'rect': [3, len(all_colors)]},
        lines_art={'linestyle': ['-', '--'], 'linewidth': 2},
        circle_art={'alpha': 0.4})

    # Creating logic circuit for zoomed figure.

    if zoom:

        # Creating labels for for boxes in figure.
        ax2.set_xticklabels([r'{%s}' % (i) for i in latex_str2],
                            fontsize=fontsize[2])

        ax2.set_title(r'\textbf{%s}' % (latex_str[0]), fontsize=fontsize[0])
        ax2.set_xlabel(r'\textbf{%s}' % (latex_str[1]), fontsize=fontsize[1])
        ax2.set_ylabel(r'\textbf{%s}' % (latex_str[2]), fontsize=fontsize[1])

        # Zooming in at y-axis via calculation.
        y_axis = ax2.get_ylim()
        axis_zoomed = (100.0 - zoom_procent[0]) / 100.0
        axis_zoomed2 = (100.0 - zoom_procent[1]) / 100.0
        y_axis_scale = y_axis[1] * axis_zoomed
        y_axis_scale2 = y_axis[0] * axis_zoomed2

        # Setting new axis for figure.
        ax2.set_ylim(top=y_axis_scale, bottom=y_axis_scale2)

        ax2.legend(
            [org_leg_objs[i] for i in range(len(org_leg_objs))],
            [r'\textbf{%s}' % (org_labels[i]) for i in range(len(org_labels))],
            numpoints=n, frameon=frameon,
            loc=loc, fontsize=fontsize[2])
        # Setting color scheme to box.
        for patch, color in zip(boxplot2['boxes'], picked_color):
            patch.set_facecolor(color)

        # Setting color for medians.
        for median in boxplot2['medians']:
            median.set(color=flier_c[1], linewidth=2)

        # Setting cap color for whiskers
        for cap in boxplot2['caps']:
            cap.set(color=flier_c[2], linewidth=2)

        # Setting color and width for whiskers
        for whisker in boxplot2['whiskers']:
            whisker.set(color=flier_c[2], linewidth=2)

        # Setting outliers colored.
        for flier in boxplot2['fliers']:
            flier.set(marker='o', color=flier_c[0], alpha=0.4)

    # Setting color scheme to box.
    for patch, color in zip(boxplot1['boxes'], picked_color):
        patch.set_facecolor(color)

    # Setting color for medians.
    for median in boxplot1['medians']:
        median.set(color=flier_c[1], linewidth=2)

    # Setting cap color for whiskers
    for cap in boxplot1['caps']:
        cap.set(color=flier_c[2], linewidth=2)

    # Setting color and width for whiskers
    for whisker in boxplot1['whiskers']:
        whisker.set(color=flier_c[2], linewidth=2)

    # Setting outliers colored.
    for flier in boxplot1['fliers']:
        flier.set(marker='o', color=flier_c[0], alpha=0.4)

    # Removing spines
    dataplotter_spines_remover(
        ax1, all_axis=False, top=False, bottom=True,
        right=False, left=True)

    dataplotter_spines_remover(
        ax2, all_axis=False, top=False, bottom=True,
        right=False, left=True)

    dataplotter_x_y_tick_remover(ax1)
    dataplotter_x_y_tick_remover(ax2)

    # Closes unwanted window.
    if zoom == False:
        plt.close(fig2)
        # Returns without zoomed option False.
        plt.show()
        return fig1, ax1

    # Returns with zoomed option True.
    plt.show()
    return fig1, fig2, ax1, ax2


def dataplotter_line(
        df, figlabels=['Untitled', 'X-axis', 'Y-axis'],
        line_names=[], gridlines=True):
    """Function creates a linear figure, by plotting
    pixel sum overtime for a colony.

    Parameters
    ----------
    df : pandas.DataFrame(object)
        The 'df' parameter is the input object that renders figure.
        The parameter should contain time and pixel values for
        corresponding time point!

    figlabels : list (optional)
        The parameter called 'figlabels' is list that contains string
        entries as items. The list is order as: Title, X-axis and
        Y-axis and figure rendered with function returns items as
        figure labels and title.

    lines_names : list (optional)
        The parameter called 'lines_names' is list and takes as
        items strings. The parameter is used for legend labelling
        lines rendered in figure.

    gridlines : boolean (optional)
        The parameters called 'gridlines' accepts only
        boolean values and renders figures either with
        or without grid lines.

    Returns
    -------
    fig1, ax1 : matplotlib.figure.Figure, matplotlib.axes.AxesSubplot
        The function returns 'fig1' and 'ax1' both Matplotlib objects.
        The 'fig1' contains the rendered figure.

    See Also
    --------
    dataplotter_save_figure : "fig1" return is used for save.

    dataplotter_fixing_textformat : For more information about
                                    latex rendering

    phen_o_links.dataset.dataset_copy_frame : For more information about
                                             data frame copy.

    phen_o_links.dataset.dataset_pick_columns : For more information
                                               about "split" parameter.
    """
    # Local variables
    fontsize = [12, 10, 8]
    colors = ['green', 'red', 'blue', 'yellow']
    hues = ['light']

    # Dataset handling
    df_work1 = ds.dataset_copy_frame(df)
    work_columns, index_column = ds.dataset_pick_columns(
        df_work1, split='groupby')
    # Text handling
    figtxt = dataplotter_textspacemanger(figlabels)
    line_names2 = dataplotter_textspacemanger(line_names)

    # Color handling
    colors2, hue1, hue2, hue3, hue4 = dataplotter_colorscheme(
        main=colors, hues=hues)
    figcolors = colors2 + hue1 + hue2 + hue3 + hue4
    figmaincolors = np.random.choice(figcolors, size=len(line_names) + 1)

    # Figure making
    dataplotter_fixing_textformat()
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_aspect('equal')
    ax1.set_title(r'\textbf{%s}' % (figtxt[0]), fontsize=fontsize[0])
    ax1.set_xlabel(r'\textbf{%s}' % (figtxt[1]), fontsize=fontsize[1])
    ax1.set_ylabel(r'\textbf{%s}' % (figtxt[2]), fontsize=fontsize[1])
    ax1.grid(gridlines, zorder=25)
    dataplotter_x_y_tick_remover(ax1)
    dataplotter_spines_remover(
        ax1, all_axis=False, top=False, bottom=True,
        right=False, left=True)
    ax1.set_xticks(
        np.arange(df_work1[index_column[0]].min(),
                  df_work1[index_column[0]].max() + 1, 2))

    # Figure plotting
    for i in range(len(line_names)):
        plt.semilogy(
            df_work1[index_column[0]], df_work1[work_columns[i]],
            label=line_names2[i], basey=2, color=figmaincolors[i])
        leg1 = plt.legend(loc=2, frameon=True, fontsize=fontsize[2])
    frame = leg1.get_frame()
    leg1.set_zorder(10)
    frame.set_linewidth(0)
    plt.show()
    return fig1, ax1


def dataplotter_histogram_outliers(
        df, percentage, binwidth, figlabels=['Untitled', 'X-axis', 'Y-axis'],
        bartext=['Untitled'], tabletitle=['Untitled']):
    """Takes any given pandas data frame object and returns histogram with
    outliers defined according to outliers2 function. In addition
    to histogram a table is also returned.

    Parameters
    ----------
    df : pandas.DataFrame (object)
        Input data frame object.

    percentage : float (optional)
        The parameter called 'percentage' accept only floats as input and
        ranges from 0.0 to 1.0.

    binwidth : float (optional)
        The parameter called 'binwidth' determines the bin size for
        histogram rendered.

    figlabels : list (optional)
        The parameter called 'figlabels' is list that contains string
        entries as items. The list is order as: Title, X-axis and
        Y-axis and figure rendered with function returns items as
        figure labels and title.

    bartext : list (optional)
        The parameter called 'bartext' is a list and items in list
        should be string typed.

    tabletitle : list (optional)
        The parameter called 'tabletitle' is the table title in use
        for the combine histogram.

    Returns
    -------
    fig1, ax1, fig2, ax2 :matplotlib.figure.Figure, matplotlib.axes.AxesSubplot
        The function returns two 'fig1' and 'fig2' are
        matplotlib.figure.Figure objects whereas as ax1 and ax2 are
        the subplots for respective figure. The 'fig1' return contains
        histogram and 'fig2' contains the table.

    See Also
    --------
    dataplotter_save_figure : "fig1" and "fig2" return is used for save.

    dataplotter_fixing_textformat : For more information about
                                    latex rendering

    phen_o_links.dataset.dataset_copy_frame : For more information about
                                             data frame copy.

    phen_o_links.dataset.dataset_pick_columns : For more information
                                               about "split" parameter.
    phen_o_links.dataset.dataset_outliers2 : For more information
                                            about outliers calculation parameter.
    """

    # Locale globe variables
    colors = ['green', 'blue', 'black', 'light', 'dark']
    fontsize = [16, 12, 10]
    fontsize = [fontsize[i] * 1.5 for i in range(len(fontsize))]
    number = 5

    # Text rendering formatting latex
    figlabels = dataplotter_textspacemanger(figlabels)
    datatext = dataplotter_textspacemanger(bartext)

    # Colorscheme rendering
    maincolor, light, light_dark, dark_light, dark = dataplotter_colorscheme(
        main=colors[:3], hues=colors[3:])
    colorscheme = maincolor + light + light_dark + dark_light + dark
    colorscheme.pop(colorscheme.index(None))
    picked_color = np.random.choice(colorscheme, 1)

    # Working with data frame
    df_work1 = ds.dataset_copy_frame(df)

    y, subset, x, x1, x2, x3, x4 = ds.dataset_outliers_midpoint(
        df_work1, percentage, number)
    print subset

    # Deleting unwanted returns
    del y, x, x1, x2, x3, x4

    # Calculates the proper bin size for subset that are true 'outliers'

    columnames, indexnames, edgevalues, bins = ds.dataset_bins_calculator(
        subset, binwidth)

    # Creating complementary table
    table = pd.DataFrame({'Original data size': len(df_work1),
                          'Outliers in data': len(subset),
                          'Counts': 'Counts'},
                         index=['Counts'],
                         columns=['Counts', 'Original data size',
                                  'Outliers in data'])
    # Creating figure with table
    fig2, ax2 = dataplotter_object_table_maker(table, figtext=tabletitle)

    # Creating figure content
    dataplotter_fixing_textformat()
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.hist(subset[columnames[0]].values, bins=bins,
             color=picked_color[0], label=r"{%s}" % (datatext[0]),
             alpha=0.35, normed=0,
             weights=np.zeros_like(
                 subset[columnames[0]].values) + 1.0 / len(
                     subset))
    ax1.set_title(r'\textbf{%s}' % (figlabels[0]), fontsize=fontsize[0])
    ax1.set_xlabel(r'\textbf{%s}' % (figlabels[1]), fontsize=fontsize[1])
    ax1.set_ylabel(r'\textbf{%s}' % (figlabels[2]), fontsize=fontsize[1])
    dataplotter_x_y_tick_remover(ax1)
    dataplotter_spines_remover(
        ax1, all_axis=False, top=False, bottom=True,
        right=False, left=True)
    ax1.legend(loc=2, frameon=False, fontsize=fontsize[2])
    yaxis_formatter = plt.FuncFormatter(dataplotter_yaxis_percenatage)
    ax1.yaxis.set_major_formatter(yaxis_formatter)
    plt.show()
    return fig1, ax1, fig2, ax2


def dataplotter_histogram(
        df, binwidth, figlabels=['Untitled', 'X-axis', 'Y-axis'],
        bartext=['Untitled'], loc=(1.05,0.8),
        atxt=False, df_atxt=[]):
    """Function renders histograms and has an option of
    highlighting sub sample from larger data frame.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' is a data frame given by
        user.

    binwidth : float or int (optional)
        The 'binwidth' determines the size
        for each representative bin given in the histogram.

    figlabels : list(optional)
        The parameter called 'figlabels' is list where items are
        string typed. The list determines title, x and y axis labelling
        by list indices e.g the first item determines title, second x-axis
        label and so on.

    bartext : list(optional)
        The parameter called 'bartext'
        sets a label for data displayed in figure. The 'bartext'
        is visible if atxt is set to False.

    atxt : boolean(optional)
        The parameter called 'atxt' is boolean, which is set to False
        at default. If change to True the histogram function renders
        a figure with highlights for a subset of points from 'df'
        given in 'df_atxt'.

    df_atxt : list(optional)
        The parameter called 'df_atxt' is list with a length of 1
        and item passed is pandas.core.frame.DataFrame.
        The 'df_atxt' is sub sampling from 'df'.

    Returns
    -------
    fig1, ax1 : matplotlib.figure.Figure, matplotlib.axes.AxesSubplot
        The return called 'fig1' contains rendered figure and
        'ax1' is it's subplot.

    """
    # Locale globe variables
    colors = [
        'green', 'red', 'yellow', 'blue', 'black', 'lightdark',
        'darklight', 'light', 'dark']
    fontsize = [16, 12, 10]
    fontsize = [fontsize[i] * 1.5 for i in range(len(fontsize))]
    alpha2 = 1.0/len(bartext)

    # Text rendering formatting latex
    figlabels = dataplotter_textspacemanger(figlabels)
    datatext = dataplotter_textspacemanger(bartext)

    # Colorscheme rendering
    maincolor, light, light_dark, dark_light, dark = dataplotter_colorscheme(
        main=colors[:-4], hues=colors[-4:])
    colorscheme = maincolor + light + light_dark + dark_light + dark

    # Get rid of color transparency
    if atxt:
        alpha2 = 1.0

    # Working with data frame
    df_work1 = ds.dataset_copy_frame(df)
    columnames, indexnames, edgevalues, calc_bins = ds.dataset_bins_calculator(
        df_work1, binwidth)

    # Getting rid of None values
    for i in range(colorscheme.count(None)):
        colorscheme.pop(colorscheme.index(None))

    picked_color = []
    while len(picked_color) < len(columnames):
        picked_color = np.random.choice(colorscheme, len(columnames))
        unique_colors = np.unique(picked_color)
        picked_color = unique_colors

    for i in picked_color:
        colorscheme.pop(colorscheme.index(i))

    print ("This is the amount the bins: {0}").format(calc_bins)
    # Creating figure conten0t
    dataplotter_fixing_textformat()
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    for i in range(len(columnames)):
        n, bins, patches = ax1.hist(
            df_work1[columnames[i]].values, bins=len(calc_bins),
            color=[picked_color[i]], label=r"{%s}" % (datatext[i]),
            alpha=alpha2, normed=0, histtype='bar', zorder=np.negative(i),
            weights=np.zeros_like(
                df_work1[columnames[i]].values) + 1.0 / len(
                    df_work1))
    ax1.set_title(r'\textbf{%s}' % (figlabels[0]), fontsize=fontsize[0])
    ax1.set_xlabel(r'\textbf{%s}' % (figlabels[1]), fontsize=fontsize[1])
    ax1.set_ylabel(r'\textbf{%s}' % (figlabels[2]), fontsize=fontsize[1])
    if atxt:
        df_sub = df_atxt[0]
        dataplotter_text_annotation_hist(
            df_sub=df_sub, patches=patches, colorscheme=colorscheme,
            bins=bins)

    dataplotter_x_y_tick_remover(ax1)
    dataplotter_spines_remover(
        ax1, all_axis=False, top=False, bottom=True,
        right=False, left=True)
    if not atxt:
        ax1.legend(loc=loc, frameon=False, fontsize=fontsize[2])
    yaxis_formatter = plt.FuncFormatter(dataplotter_yaxis_percenatage)
    ax1.yaxis.set_major_formatter(yaxis_formatter)
    plt.show()
    if atxt:
        print ("\nDon't forget to save figure with lengend_tight "
               "set as True and add the ax1 return to "
               "axes_container for dataplotter_save_figure "
               "function!!")
    return fig1, ax1


def dataplotter_plate_position_scoring_heatmap(
        plate_coordinates, title=['Untitled']):
    """ The function returns a schematic plate
    with 1536 position. The function returns position
    enrichments.

    Parameters
    ----------
    plate_coordinates : list
        The parameter called 'plate_coordinates' is list with tuple
        pairs for row and column. Valid entries for row ranges
        from 0-31 and for column 0-47.

    title : list (optional)
        The parameter called 'title' renders the title for figure.

    Returns
    -------
    fig1, ax1 : matplotlib.figure.Figure, matplotlib.axes.AxesSubplot

        The return called 'fig1' contains rendered figure and
        'ax1' is it's subplot.
    """

    # Local Globals
    figuretext = title + ['Column Coordinates',
                          'Row Coordinates', 'Data Frequencies']
    fontsize = [14, 12, 10]

    # Used for labelling position at plate
    rows = range(0, 32)
    columns = range(0, 48)

    # Generating plate
    plate_matrix = np.zeros((32, 48))

    # Function calls
    plate_matrix2 = ds.dataset_plate_position_scoring(plate_matrix,
                                                      plate_coordinates)

    figtext = dataplotter_textspacemanger(figuretext)

    # Rendering Figure
    dataplotter_fixing_textformat()
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(111)

    # Creating heat map data
    heat_map = ax1.matshow(plate_matrix2, interpolation='nearest')

    # Adding color bar
    colorbar = fig1.colorbar(heat_map)

    # Adding text and fixing coordinates
    ax1.set_title(r'\textbf{%s}' % (figtext[0]), fontsize=fontsize[0])
    ax1.set_xlabel(r'\textbf{%s}' % (figtext[1]), fontsize=fontsize[1])
    ax1.set_ylabel(r'\textbf{%s}' % (figtext[2]), fontsize=fontsize[1])
    colorbar.set_label(r'\textbf{%s}' % (figtext[3]), fontsize=fontsize[1])
    ax1.set_xticks(np.arange(plate_matrix2.shape[1]))
    ax1.set_yticks(np.arange(plate_matrix2.shape[0]))
    ax1.set_yticklabels(rows)
    ax1.set_xticklabels(columns)
    fig1.set_size_inches(8, 6)
    ax1.set_aspect('equal')
    windowmanger = plt.get_current_fig_manager()
    windowmanger.resize(* windowmanger.window.maxsize())
    plt.show()

    return fig1, ax1


def dataplotter_hierarchical_clustered_heatmap(
        df, cutoff_percentage=None, algorithm='complete', norm_color=None,
        heatmap_setting=['nearest', 'auto', 'lower'],
        figure_text=['Title', 'X-label', 'Distance', 'Data'],
        a_text=[False, False], a_genes=[],rotate=90):
    """ The function returns a hierarchical clustered heat map
    from a pandas data frame object, which has been indexed.
    Data frame should have the shape of 2 columns and multiple
    rows indices.

    Parameters
    ----------
    df : pandas.DataFrame(object)
        The 'df' parameter is a pandas data frame object
        that has been indexed by certain column or values
        before.
    cutoff_percentage : float(optional)
        The parameter 'cutoff_percentage' is utilized as a to set threshold for
        where the hierarchical tree is cut to clades.

    algorithm : str(optional)
        The parameter called "algorithm" is set to 'complete', which
        determines the cluster algorithm used for clustering
        of data. To see more valid string entries search for
        scipy.cluster.hierarchy.linkage.

    norm_color : boolean(optional)
        The parameter called "norm_color" normalizes the color rendering
        for figure and colorbar between the negative and positive value
        of absolute max value. The 'norm_color' is set to 'None' and
        if change to 'True' figure is rendered with color normalization.

    heatmap_setting : list(optional)
        The 'heatmap_setting' parameter is a list object that consist of
        3 items. The all entries are string based and the ith-order
        has different properties.
        The 'heatmap_setting' = ['Interpolation method used',
        'Aspect for heat map', 'Where first element is rendered'].
        For more information look up documentation for imshow in
        Matplotlib homepage or other source.

    figure_text : list(optional)
        The parameter called 'figure_text' contains all descriptive text
        for figure e.g. axis labels, figure title etc..

        The "figure_text" consists of 4 string entries, where the ith-order
        determines different properties.

        "figure_text" = ["Figure Title", "X-axis Label",
        "Y-axis Label", "Colorbar Label"].

    a_text : list boolean(optional)
        The parameter called 'a_text' is abbreviation from annotation text and
        it consist of 2 list items. The both items are set to "False" and the
        ith order results in different annotations styles, which are
        suited for different purposes. The 1st being all data points
        labelled works on dataset <= 50 points.
        a_text=["All data is labelled", "Subset of data is labelled"]

    a_genes : list(optional)
        The parameter 'a_genes' is list object,
        contains string values found from index present in "df".

    Returns
    -------
    fig1 : matplotlib.figure.Figure(object)
        The 'fig1' is the object that contains the hierarchical cluster.

    df_cluster : pandas.core.frame.DataFrame(object)
        The 'df_cluster' is the input 'df' with the clades.
    """
    # Global Local
    fontsize = [18, 12]

    # Check that logic
    if a_text[1]:
        if not a_genes:
            print "\n\nList 'a_gene' empty not valid option.\n\n "
            return ValueError

    # Copy frame.
    df_copy = ds.dataset_copy_frame(df)

    # Sub sampling
    work, index = ds.dataset_pick_columns(df_copy, split='groupby')

    df_work1 = df_copy[index + work]


    df_work1.columns = [i.replace('_',' ') for i in df_work1.columns]

    df_work1 = df_work1.set_index(df_work1.columns[0])

    # Check that format is valid

    if not(df_work1.shape[1] == 2):
        propmt = (' Subset given does not follow shape need '
                  ' for heat-map figure. 3 columns in:'
                  ' Indexer and 2 work columns')
        raise ValueError(propmt)
        return

    vmin, vmax = ds.dataset_normalise_values_heatmap(df_work1)
    # Calculation of square distances.
    row_distance, column_distance = ds.dataset_pairwise_square_distance_heatmap(df_work1)
    # Clustering the square distances.
    cluster = ds.dataset_cluster_data_heatmap(
        data=[row_distance, column_distance], calc_method=algorithm)

    # Calculating the Cophenetic correlation
    c, coph_dist = sch.cophenet(cluster[0], distance.pdist(df_work1))
    c = np.round(c, decimals=4, out=None)

    # Calculating color threshold for dendrogram
    if not cutoff_percentage:
        cutoff_percentage = 0.5
    cutoff = np.max(cluster[0][:, 2]) * cutoff_percentage

    # Heat map colors normed to max and min of input data.
    if norm_color is True:
        norm_color = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Fixing fonts
    dataplotter_fixing_textformat()
    figure_text = dataplotter_textspacemanger(figure_text)

    # Creating figure object.
    fig1 = plt.figure(figsize=(8, 6))

    # Creating a 2x2 grid where subplots are rendered.
    figure_layout = gridspec.GridSpec(
        2, 2, wspace=0.0, hspace=0.0, width_ratios=[0.25, 1.0],
        height_ratios=[0.25, 1.0])

    # Top hierarchical dendrogram  cluster.
    col_denD_AX = fig1.add_subplot(figure_layout[0, 1])
    col_denD = dataplotter_dendrogram_maker(cluster[1], cutoff=cutoff)
    # Removing ticks and spines from axes.
    dataplotter_x_y_tick_remover(col_denD_AX, all_ticks=True)
    dataplotter_spines_remover(col_denD_AX)

    # Right hierarchical dendrogram cluster.
    row_denD_AX = fig1.add_subplot(figure_layout[1, 0])
    row_denD_AX.axvline(x=cutoff, ymax=1, c='k', ls='--', lw=1.5)
    row_denD = dataplotter_dendrogram_maker(cluster[0], pos='left',
                                            cutoff=cutoff)
    # Removing ticks and spines from axes.
    dataplotter_x_y_tick_remover(row_denD_AX, all_ticks=False, x=True)
    dataplotter_spines_remover(row_denD_AX)

    # Heat map over data ordered as dendrogram.
    heatmap_AX = fig1.add_subplot(figure_layout[1, 1])
    heatmap_fig = heatmap_AX.imshow(
        df_work1.iloc[row_denD['leaves'], col_denD['leaves']],
        interpolation=heatmap_setting[0], aspect=heatmap_setting[1],
        origin=heatmap_setting[2], norm=norm_color, cmap=plt.cm.RdBu)
    # Removing ticks and spines from axes.
    dataplotter_x_y_tick_remover(heatmap_AX, all_ticks=True)
    dataplotter_spines_remover(heatmap_AX)

    if a_text[0] == False and a_text[1] == False:
        # Column labels
        heatmap_AX.set_xticks(np.arange(df_work1.shape[1]))
        col_labels = heatmap_AX.set_xticklabels(
            df_work1.columns[col_denD['leaves']])
        # Rotation of column labels
        for col_label in col_labels:
            col_label.set_rotation(rotate)

    # Adding labels to data
    if a_text[0]:
        if len(df_work1.index) >= 1000 and a_text[1] == False:
            print "\n\nThe figure might take a long while to be rendered!\n\n"
        # Row labels
        heatmap_AX.set_yticks(np.arange(df_work1.shape[0]))
        heatmap_AX.yaxis.set_ticks_position('right')
        heatmap_AX.set_yticklabels(
            df_work1.index[row_denD['leaves']])
        # Column labels
        heatmap_AX.set_xticks(np.arange(df_work1.shape[1]))
        col_labels = heatmap_AX.set_xticklabels(
            df_work1.columns[col_denD['leaves']])
        # Rotation of column labels
        for col_label in col_labels:
            col_label.set_rotation(rotate)
        # Removing tick lines
        for lines in heatmap_AX.get_xticklines() + heatmap_AX.get_yticklines():
            lines.set_markersize(0)

        if a_text[1]:
            y_ax_instances = heatmap_AX.yaxis.get_majorticklabels()
            y_labels = [
                y_ax_instances[i].get_text() for i in range(
                    len(y_ax_instances))]
            y_indicies = range(len(y_ax_instances))
            y_coords = pd.DataFrame(
                {'Strain': y_labels, 'x': 1, 'y': y_indicies})
            y_sub_labels = y_coords[y_coords.Strain.isin(a_genes)]
            heatmap_AX.set_yticklabels([])
            #bbox_props = dict(boxstyle="larrow, pad=0.3", fc="white")
            arrows_props = dict(arrowstyle="->", connectionstyle="arc3")
            for i in range(len(y_sub_labels)):
                y_offset = [15, 60, -15, -60]
                y_random = np.random.choice(
                    y_offset, 1, p=[0.375, 0.125, 0.125, 0.375])

                plt.annotate(
                    y_sub_labels.Strain.values[i],
                    xy=(1, float(y_sub_labels.y.values[i]) / len(y_labels)),
                    xycoords=heatmap_AX, xytext=(30, y_random),
                    textcoords="offset points", arrowprops=arrows_props)

    # Adding Figure descriptives
    text = 'Cophenetic Correlation Coefficient = '
    col_denD_AX.figure.suptitle(
        r'\textbf{%s}' % (figure_text[0]), fontsize=fontsize[0], x=0.75)
    heatmap_AX.set_xlabel(
        r'\textbf{%s}' % (figure_text[1]), fontsize=fontsize[1])
    row_denD_AX.set_ylabel(
        r'\textbf{%s}' % (text + str(c)), fontsize=fontsize[1])
    row_denD_AX.set_xlabel(
        r'\textbf{%s}' % (figure_text[2]), fontsize=fontsize[1])

    # Creating colorbar from  data input.
    colobar_layout = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=figure_layout[0, 0], wspace=0.05,
        hspace=0.05)
    colorbar_AX = fig1.add_subplot(colobar_layout[0, 1])
    colorbar = fig1.colorbar(heatmap_fig, colorbar_AX)

    # Creating color bar ticks.
    n_minus = np.linspace(np.floor(vmin), 0, abs(np.floor(vmin) - 1))
    n_plus = np.linspace(np.ceil(vmax), 0, abs(np.ceil(vmax) + 1))
    n_ticks = np.hstack((n_minus, n_plus))
    n_ticks = np.unique(n_ticks)

    # Setting color bar ticks.
    colorbar.set_ticks(n_ticks)
    colorbar.set_label(figure_text[-1])
    colorbar.ax.yaxis.set_ticks_position('left')
    colorbar.ax.yaxis.set_label_position('left')
    colorbar.outline.set_linewidth(0)
    ticklabels = colorbar.ax.yaxis.get_ticklabels()

    # Formatting text layout.
    for text in ticklabels:
        text.set_fontsize(text.get_fontsize() - 3)

    # Display figure
    plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    # Getting clusters
    df_cluster = df_work1.iloc[row_denD['leaves'], col_denD['leaves']]
    groups = sch.fcluster(cluster[0], cutoff, criterion='distance')
    df_cluster['Id_groups'] = groups

    # Saving file
    df_cluster.columns = [i.replace(' ', '_') for i in df_cluster.columns]
    name = 'Hierachical_Clustered_with_cutoff_' + str(cutoff)

    ds.dataset_filesave(df_cluster, filename=name)

    return fig1, df_cluster


def dataplotter_kde_plot(
        df, filename="untitle", index=[], columns=[],
        figlabels=["Title", "X label", "Y label"],
        datalabels=["Data 1", "Data 2"], x_limits=(-1, 1)):
    """Take a given data frame and returns 2 kernel density estimates lines.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The 'df' parameter is the given pandas DataFrame (object)

    filename : str(object)
        The 'filename' is the relative path from current working directory.

    index, columns : list(object)
        The parameter called 'index' and 'columns' are specify data input.
        The 'index' groups the 'df' in nth amount of indices and the
        'columns' parameter specify the 2 labels with values.

    figlabels : list(object)
        The parameter called 'figlabels' is list object with the
        length of 3 items. The order of the items determines different
        features. 1th item is title, 2nd item is xlabel, 3rd item is ylabel.

    datalabels : list(object)
        The parameter called 'datalabels' has length of 2 items
        which corresponds to the data inputs given in 'columns'.

    x_limits : tuple(object)
        The parameter called 'x_limits' determines the x-axis range
        form x-max to x-minimum value.

    Returns
    -------
    svg : figure(object)

        The function returns nth amount of figure at the current
        working directory.

    Raises
    ------
    ValueError
        If 'columns' or 'index' items not found in main 'df'.
        If 'filename' is not a string object.
    IndexError
        If 'index' given to group 'df' is not divisible by 2.

    See Also
    --------
    phen_o_links.dataset.dataset_pick_columns : For more information about
                                                function called when 'index'
                                                and 'columns' is empty.
    """

    # Global Local
    indices_key = []

    # Copy frame
    df1 = ds.dataset_copy_frame(df)

    figtext = dataplotter_textspacemanger(figlabels)
    datatext = dataplotter_textspacemanger(datalabels)

    # Function call
    if not(index and columns):
        columns, index = ds.dataset_pick_columns(df1, split="groupby")

    if not isinstance(filename, str):
        text = ("Make sure that 'filename' parameter is a string object."
                " User input was {0}").format(filename)
        raise ValueError(text)

    if index and columns:
        var = index + columns
        df1_columns = df1.columns.tolist()
        test = [i for i in var if i not in df1_columns]
        if test:
            text = ("Items given in 'columns' are not found "
                    "User input {0}").format(var)
            raise ValueError(text)

    # Creating pandas object thats is grouped by index
    df1_gr = df1.groupby(index)
    indices_key = df1_gr.indices.keys()

    if len(indices_key) % 2:
        text = ("This function is viable if indices for a given index "
                "is divisible with 2. The length of indices given by "
                "'index' parameter is {0}").format(len(indices_key))
        raise IndexError(text)

    # Loop that plots stuff
    for i in indices_key[::2]:
        nr = indices_key[i-1:i+1]
        for d in nr:
            df1_gr.get_group(d)[columns[0]].plot(
                kind="kde", ls='--', lw=2.5,
                label=r"%s" % (datatext[0] + str(d)))
            df1_gr.get_group(d)[columns[1]].plot(
                kind="kde", lw=2.5,
                label=r"%s" % (datatext[1] + str(d)))
        plt.xlim(x_limits)
        plt.axvline(x=0, color="black", ls='--', label=r"Origin", lw=1.0)
        plt.title(r"%s" % (figtext[0]))
        plt.xlabel(r"%s" % (figtext[1]))
        plt.ylabel(r"%s" % (figtext[2]))
        plt.legend(loc=(1.02, 0.75), frameon=False)
        plt.tick_params(axis="x", which="both", top="off")
        plt.tick_params(axis="y", which="both", right="off")
        plt.savefig(filename+str(i-1)+"_"+str(i+1)+".svg",format="svg")
        plt.clf()
    return "Figures are done"


def dataplotter_kde_six_sigmas_cutoff(
    df, six_sigmas, obs_column=[""], filename="untitled",
    figlabels=["Untitle", "X axis", "Y axis"], datalabel=["Data Observed"],
    xlimits=(-1,1)):
    """
    The function renders kernel density plot over observed data with sigmas
    cutoffs from a given null hypothesis. The null hypothesis are labelled with
    subscript empty set symbol(see google search latex emptyset).

    Parameters
    ----------
    df : pandas.core.frame.DataFrame(object)
        The "df" parameter is a pandas data frame that contain the label
        with the observed data.

    six_sigmas : pandas.core.frame.DataFrame(object)
        The 'six_sigmas' is a pandas data frame returned from function call
        ds.dataset_six_sigmas_cutoff. The frame contains the sigmas values
        produced in figure.

    obs_column : list(object)
        The "obs_column" contains the column label of the observed data.
        The obs_column accepts only strings as item and has length of 1.

    filename : str(optional)
        The 'filename' parameter is the relative path from current working
        directory. The parameter both points to location of saving file and
        it's the name of the file.

    figlabels, datalabel : list(optinal)
        The 'figlabels' and 'datalabel' are the figure annotation text.
        Both parameters are accepts only strings as input. The input order for
        'figlabels' matters! Frist item is figure title, second item is x-axis
        label and last entry is y axis label. The parameter 'datalabel' is the
        label used for 'observed data' in figure.

    xlimits : tuple(object)
        The parameters 'xlimits' is tuple object, which determines the x-axis
        range of the figure.

    Returns
    svg : file
        The function returns a svg file in the end.

    Raises
    ------
    ValueError
        If 'filename' and 'obs_column' are not string typed.
        If 'obs_column' is not found or length is not equal to one.
        If 'obs_column' label and 'six_sigma' label is not present as a column
        label.

    See Also
    --------
    phen_o_links.dataset_six_sigmas_cutoff : For more information about
                                             'six_sigmas' data frame.

    """
    # Local Global
    datatext = []
    figtext = []
    colors = ['green', 'red', 'yellow', 'colorblind']
    shades = ['light', 'darklight', 'dark', 'lightdark']
    mean_dict = {0:r"Mean$_{\emptyset}$"}
    no_value = r" x sigmas$_{\emptyset}$"
    colors2 = []

    # Copy frame
    df1 = ds.dataset_copy_frame(df)

    # Checking input
    if not(isinstance(filename,str)):
        text = ("{0} not a string typed").format(filename)
        raise ValueError(text)
    if not (len(obs_column)==1 and isinstance(obs_column[0],str)):
        text = ("{0} not a string typed or length is not").format(obs_column)
        raise ValueError(text)

    # Checking column labels
    c1 = obs_column[0] in df1.columns
    c2 = "Null_mean" in six_sigmas.columns

    if not(c1 and c2):
        text = ("Label input in 'obs_column' found: {0}"
                "\nLabel input in 'six_sigmas' found : {1}"
                "").format(c1,c2)
        raise ValueError

    # Creating color pallet
    m,l, ld, dl, d = dataplotter_colorscheme(main=colors, hues=shades)
    l_colors = m + l + ld + dl + d
    l_colors = [i for i in l_colors if i]

    # Picking colors
    while len(six_sigmas) > len(colors2):
        picked_color = set(np.random.choice(l_colors, len(six_sigmas)))
        colors2 = picked_color

    # Creating list with colors
    colors2 = list(colors2)
    # Fixing text
    figtext = dataplotter_textspacemanger(figlabels)
    datatext = dataplotter_textspacemanger(datalabel)
    mu_null = str(np.around(six_sigmas.Null_mean[0], 4))
    std_null = str(np.around(six_sigmas.Null_std[0], 4))
    mean_dict[0] = r"%s" %("Mean$_{\emptyset}$ = " + mu_null)

    # Plotting data
    df1[obs_column[0]].plot(
        kind="kde", lw=2.5, c="#0C5F83", ls="-", label=r"%s" % (datatext[0]))
    # Adding std null to legend
    plt.axvline(x=0, c="black", ls="-.", lw=0,
                label=r"%s" % (r"std$_{\emptyset}$ =\ "+std_null))

    # Adding cutoffs from six sigmas frame
    for i in range(len(six_sigmas)):
        plt.axvline(
            x=six_sigmas.Left.values[i], lw=2.5, ls="--", c=colors2[i])
        plt.axvline(
            x=six_sigmas.Right.values[i], ls="--",lw=2.5,
            c=colors2[i], label=r"%s" % (mean_dict.get(
                i,str(six_sigmas.Null_Sigmas.values[i]) + no_value)))

    # Adding figure text and saving!
    plt.xlim(xlimits)
    plt.title(r"%s" % (figtext[0]))
    plt.xlabel(r"%s" % (figtext[1]))
    plt.ylabel(r"%s" % (figtext[2]))
    plt.legend(loc=(1.02, 0.5), frameon=False)
    plt.tick_params(axis="x", which="both", top="off")
    plt.tick_params(axis="y", which="both", right="off")
    plt.savefig(filename+".svg",format="svg")
    plt.clf()
    plt.close("all")
    return "Figure with sigmas cut offs is done"


def dataplotter_go_colors(go_table):
    """ Takes a single data frame with and returns a dict with colors
    for found GO Slim Terms.

    go_table : pandas.core.frame.DataFrame(object)
        The parameter called 'go_table' must contain a column labelled
        'GO_Slim_Term'.

    Returns
    -------
    go_colors : dict(object)
        The 'go_colors' is dictionary where the keys are the GO slim terms
        and the value is a color. The 'go_colors' is global variable.

    See Also
    --------
    dataplotter_go_enrichment_plot : For the usage of 'go_colors'.
    """
    # Copying frame
    go_t = ds.dataset_copy_frame(go_table)
    # Global variable
    global go_colors
    # Local Globals
    sorted_by_names = []
    rm_color = ["white"]
    rm_color2 = ["ivory", "aliceblue", "mintcream", "azure", "snow", "w"]

    # Adding creating colors
    all_colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS)

    # Sorting colors by hue, saturation, value and as well as the name
    by_hsv = sorted((
        tuple(mpl.colors.rgb_to_hsv(mpl.colors.to_rgba(color)[:3])), name)
        for name, color in all_colors.items())

    # Get the sorted color names.
    picked_colors = [name for hsv, name in by_hsv]

    print ("Number of colors before filtering"
           " out unwanted: {0}").format(len(picked_colors))

    # Pick out colors that have word white.
    rm_white = [i for i in picked_colors if rm_color[0] in i]

    # Adding unwanted colors together
    rm_colors = rm_white + rm_color2

    # Removing unwanted colors from main colors
    [picked_colors.pop(picked_colors.index(i)) for i in rm_colors]

    print ("Number of colors left after filtering "
           "colors {0}").format(len(picked_colors))

    # Redistributing colorscheme
    np.random.shuffle(picked_colors)

    # All unique go terms found in 'go_table'
    go_terms = go_t.GO_Slim_Term.unique().tolist()

    go_terms.sort()

    go_colors = {go_terms[i]: picked_colors[i] for i in range(len(go_terms))}
    return go_colors


def dataplotter_go_enrichment_plot(
    df, figtitle=["Untitled"], filename="untitled", path_to_save="./"):
    """Takes the return files from phen_o_links.dataset_go_enrichment and
    returns a horizontal bar plot for GO slim terms that passed FDR.
    The bar plot is saved as an svg image.

    Parameters
    ----------

    df : pandas.core.frame.DataFrame(object)
        The parameter called 'df' is a imported '.csv'-file created from
        'phen_o_links'.dataset_go_enrichment return.

    figtitle : list(optional)
        The parameter called 'figtitle' is list object with a length of 1.
        The parameter accepts strings.

    filename : str(optional)
        The 'filename' is the name of saved svg file created by function.
        The default value of 'filename' is set to 'untitled'.

    path_to_save : str(optional)
        The 'path_to_save' is the relative of absolute path from current
        working directory. The 'path_to_save' default value is './'.

    Returns
    fig1, ax1 : matplotlib.pyplot(objects)
        The 'fig1' return is the figure object of the plot and the 'ax1' is
        the axes object of the figure.

    Raises
    ------
    ValueError
        If global variable called 'go_colors' is empty.

    See Also
    --------
    dataplotter_go_colors : For more information about 'go_colors' empty.
    dataplotter_save_figure : For other saving option of 'fig1' and 'ax1'
                              returns.
    phen_o_links.dataset_go_enrichment : For more information about 'df' input.

    """
    # Global variable
    global go_colors

    # Copying frame
    df1 = ds.dataset_copy_frame(df)

    # Local globals
    x_label = r"%s" %("Enrichment log$_{2}$")
    labels = ["Dataset_vs_Database", "Sub_vs_Database", "Sub_vs_Dataset"]
    names_switch =  {i: "Enrichment" for i in labels}
    go_terms = [i for i in df1.columns if "Slim" in i]
    df1 = df1.rename(columns={i:names_switch.get(i,i) for i in df1.columns})
    org_columns = df1.columns.tolist()
    interval = df1.Interval.unique().astype(list)[0]
    title_suffix = r"\newline for sigma$_{\emptyset}$ interval$_{%s}$." %(interval)
    figtitle2 = dataplotter_textspacemanger(figtitle)
    file_to_save = path_to_save + filename + ".svg"

    # Making sure that user has colors input.
    if not go_colors:
        text =("Please run function called dataplotter_go_colors"
               " before plotting go slim enrichments!")
        raise ValueError(text)

    # Creating colors for all bars
    bar_colors = [go_colors.get(i) for i in df1[go_terms[0]].tolist()]

    # text fixing formatting latex rendering
    columns_rm_underscore = dataplotter_textspacemanger(
        org_columns, pattern="_", output=" ")

    columns = dataplotter_textspacemanger(columns_rm_underscore)

    # Changing column names!
    df1.columns = columns

    # Indexing df1 frame
    df1 = df1.set_index([i for i in df1.columns if "Slim" in i])

    # Making it GO terms latex formatted
    index_names = df1.index.tolist()
    index_names2 = dataplotter_textspacemanger(
        index_names, pattern="_", output=" ")
    index2 = dataplotter_textspacemanger(index_names2)

    index_dict = {
        index_names[i]:r"%s" %(index2[i]) for i in range(len(index_names))}
    df1 = df1.rename(index=index_dict)

    # Creating canvas and axes
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(111)

    # Plotting horizontal bar plots

    df1[df1.loc[:, "FDR"] == True].Enrichment.plot(
        kind="barh", ax=ax1, color=bar_colors)

    # Removing spines
    dataplotter_spines_remover(
        ax1, left=True, bottom=True, right=False, top=False, all_axis=False)

    # Fixing edge colors
    prp_axis = ax1.properties()

    nr_bars = [i for i in  range(len(ax1.yaxis.get_majorticklabels()))]
    bars_prp = prp_axis.get("children")

    for i in bars_prp[:len(nr_bars)]:
        i.set_edgecolor("black")
    fig1.subplots_adjust(left=0.48)

    # Setting text
    ax1.set_xlabel(x_label)
    fig1.suptitle(r"%s" %(figtitle2[0]+title_suffix))

    # Saving figure
    plt.savefig(file_to_save, format="svg")

    return fig1, ax1



if __name__ == "__main__":
    # Execute only as script
    print "Please import module named {0} with Ipython".format(__name__)


