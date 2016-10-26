Project Summary
===============

The aim of the script is:

    Compare phenotypes.
    Make graphs.
    Highlight interesting data or outcomes.


Needed Steps
____________

Need to find type of file structure and some sort file system.

Data must be cleansed from inf or NAN values.

Data must be analysed with statistical tools

Data must be plotted and data points must be referred or
highlighted.


Data_plotter.py fixes
_____________________

Check if data is normalised distributed

Do this via pandas and matplotlib his source code

Check if there are ways to weigh several data points into groups



Goals Achieved
______________

File structure chosen was tab separated vector, also known as 
tsv files. Files are manipulated, loaded, or created via python module
pandas version 0.13. Files were all saved with fixed structure of
columns. Suffix for files was csv and file names are given by author.


Nan values and inf are taken care of with help of from pandas and numpy
module, via help of python innate string module called str.replace().
However the filtering method of choice suffers from strictness and data points
with valid information are sorted out. 

Statistical issues were resolved with help from scipy via regression
trend line calculations.


Lessons Learned
_______________

Tips when creating a file system create a per of rules
that are consistent for all project e.g. all directories within a
project should have read me file that gives an overview of intrinsic files
and sub directories. These file should also include exception or
comments.

Systematic names for file system are hard to stick with and are easily 
diverted from a made up standard.In addition file names tends to lack
all the information about the file and its properties e.g. dates, media
types and genetic background are often forgotten to be part of the file
name.


Nan or inf values are hard to get rid of but the are easy to dealt with when
inf values are replaced by numpy.nan() values then its just a matter of
dropna in pandas by a subset of columns.





