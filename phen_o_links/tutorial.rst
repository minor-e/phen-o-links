Phen-o-liniks
=============

The tutorial assumes that the system in use is linux based or mac and
that the user has some previous knowledge about terminal

This section describes a quick step by step tutorial of how to use
phenolinks.py, data_assembler.py, dataplotter.py and the importance of
yeastmine.csv file.



Please make sure that all packages are installed before working
with phenolinks or other scripts.

For installation procedures please search the web. 


Requirements:
    
    Python 2.7.3 or 2.7. higher
    
    Ipython 0.12.1 or ipython qtconsole 0.12.1

    rpy2 2.3.9 version
        Package handles R object in and with python.

    dvipng 1.14 or greater

    TeX Live packages, -basic, -science, -latex, -latex-extra
        For latex packages for text conversion.
        pdfTeX 3.1415926-2.5-1.40.14 (TeX Live 2013/Debian)
        kpathsea version 6.1.1
        Copyright 2013 Peter Breitenlohner (eTeX)/Han The Thanh (pdfTeX).
        There is NO warranty.  Redistribution of this software is
        covered by the terms of both the pdfTeX copyright and
        the Lesser GNU General Public License.
        For more information about these matters, see the file
        named COPYING and the pdfTeX source.
        Primary author of pdfTeX: Peter Breitenlohner (eTeX)/Han The Thanh (pdfTeX).
        Compiled with libpng 1.2.49; using libpng 1.2.50
        Compiled with zlib 1.2.8; using zlib 1.2.8
        Compiled with poppler version 0.24.5
        
 

    Packages needed to be install are:
        
        python-pandas version 0.12.0 or higher

        python-numpy version 1.8.1 or higher

        python-scipy version 0.14.0 or higher

        python-matplotlib version 1.1.1rc or higher
        
        python-glob

        python collections


    Files needed:

        yeastmine.csv

        This file is created via:

        http://yeastmine.yeastgenome.org/yeastmine/begin.do

        and saved as csv tab delimited or tsv -file.


Pre-quested steps
_________________


Step 0

Scan-o-Matic creates tab delimited csv files, after quality
control is performed by the user. The file name is set by user and 
the suffix is relPheno.csv.


These csv-files consists of seven columns:
    
    Strain = Labelling for genetic background and mutation

    P = Fixture position from scanner ranges from 0-3

    C = Columns of plate

    R = Rows of plate

    Lag = Extracted phenotype Lag 

    Rate = Extracted phenotype Generation Time

    Yield = Extracted phenotype Yield
    
    Make sure that these columns exist in the file.
    This can either checked or added  with excel or calc 


These csv-files are restructured and exported via a
module in python called pandas.


Step 1

To optimize the process have all the relPheno.csv files in one directory
since most of the work is done interactive with ipython or ipython qtconsole.

Open an terminal in linux or mac

* For linux users running Ubuntu 12.04 LTS, press ctrl+alt+t
  and a terminal window should pop-up.


Step 2

Change directories to wanted one. Type cd and the directory name.
For more information on how to navigate in terminal and other
commands please visit this web site:

https://help.ubuntu.com/community/UsingTheTerminal


Step 3

For interaction purposes please use ipython or ipython qtconsole.

Type in command line:
    
    ipython

    or

    ipython qtconsole (recommended for novice)


Step 4

A new window has open and if the installation of the ipython qtconsole
is correct or the terminal has changed its appearance.(Don't panic !)


Step 5

These abbreviation or terms are in needed of clarification:
    * df + digit = pandas imported object often but not always 
      a DataFrame object.

    * '\\t' = blank space tab.

    * path_file = The path to the wanted file from location a to wanted
      file e.g. 'root/user/example/file_wanted.txt'.

    * var + digit = store of variable 

    * list + digit = a container for lists types

    * str + digit = plain text within single or double citation quotes.


Step 6

With in the new ipython qtconsole type the following:

import pandas as pd

This will result in a new blank line, if pandas has been properly
installed and running.

For further exploration about pandas functionality try typing 

pd. and press tab 

on the keyboard in the terminal, which results 
in list view of options for the pandas module.

Also check out pandas own homepage:

http://pandas.pydata.org/pandas-docs/stable/


Step 7

Reading in files with pandas is not a hard task, just type the following
in the ipython qtconsole terminal window.


var1 = pd.read_csv(path_file, delimiter='\\t', header=0)

press enter key

Next let's make a copy of the read in file by typing:

var2 = var1.copy()

press enter

Next lets make sure that pandas understands that var2 is an object
within pandas by typing:

df = pd.DataFrame(var2)


Step 8

Lets start manipulating the data assuming that the relPheno.csv file has
has the above mentioned columns.


Step 9

This filters away the reference colonies or control colonies from data.
Important to notice that phenolinks re_name function uses certain strain
names criteria to distinguish between reference background or other type.
This can be change to function with other types of flagging. 

To have access to script function were going to call the script via
%loadpy and the path to script.

This going to load all the functions of the scripts

%loadpy path_file  phenolinks.py

press enter key twice after that.

Step 10

After loading the phenolinks.

Type in window:

re_name(df)

This will re_name the column named Strain and get rid of the flagging
for genetic background.

Returns Strain column with out specified flag.

Step 11

The str1 works as a placeholder for the filtering value

str1 = 'for reference or control.'


Step 12

The pandas object is sliced at the Strain column via the 
filtering value stored in str1

Type the line below and press the enter key.

df2 = df[df["Strain"]!=str1]


Step 13


Sort data by plate or fixture number by typing the following line

df3 = df2[df2["P"]==digit]

The digit ranges from 0-3


Step 14

The data needs to be in alphabetical order the line below fixes that:

df4 = df3.sort('Strain')


Step 15

The df4 object contains now a plate without control strains and needs to
be saved for further processing.

df4.to_csv(path_file, sep="\\t")

Step 16

Repeat the following steps 10-15 for each fixture number in relPheno.csv
file 

Tips:

Create file names that are verbose instead of the opposite.

Getting Started with phenolinks
_______________________________
The csv-file created in step 16 are now dived by fixture number 

Step 17

Open file csv file with calc or excel and assemble an new csv file with
following columns:

Groups = Label based groups


Strain = Strain name and genetic background 


P = Fixture number or plate number ranges 0-3


C = Column in plate


R = Row in plate


Phenotype A in milieu B 


Phenotype A in milieu Basal or reference medium


Phenotype C in milieu D


Phenotype C in milieu Basal or reference medium 


Step 18

Make sure that wanted directory also contain csv-file with above format.


Start by importing phenonlinks script to wanted working directory and
then type in ipython or ipython qtconsole the following line:

%loadpy phenolinks.py


Step 19

Now import working csv-file by typing the followin lines:


var1 = pd.read_csv(path_file, sep='\\t',header=0)


var2 = var1.copy()


df = pd.DataFrame(var2)


Step 20

Data filtering and limitation with method.
This is bulky filtering system and points with valid
data will be sorted out with the method,
due to paring data condition.

Start filtering data by typing:

data_filtering(df)

The result will be shown and a new csv-file called:

* filtered_data_changeme.csv

As the title suggest this file should be re-branded and re imported
for further analysis.


Step 21

call the following function 

re_name(df)

This prompt which type of genetic background file has.


Step 22

Add GO slim terms terms by calling 

go_slim_term_create(df)

This results in two new columns called GOslimTerms and Database.


Step 23

Now we can re-flag the Strain column for genetic background via

re_flag(df)

This will ask some interactive questions.

The result will be prompted and user will be reminded to change name of
file 


step 24

Dataframe should be sorted and grouped for further analysis

use funtion call re_group()

and change name of outcome file!

Assembling Data with Phenolinks
===============================


This step is an easy and simple task.


Step 1

Make sure that all the re-flag csv files are in one directory 

and that all changme.csv file in another directory.

Be sure to not mix together reference background with experimental
background. 


Step 2

After the division is done just copy data_assembler.py 
into wanted directory. 


Step 3

cd to wanted directory with Ipython or Ipython qtconsole


Step 4

Type this in working directory:

%loadpy data_assembler.py


Step 5

Then type the following line:

    data_assembler()

The result will be prompted and reminder of changing 
the name of assembled_changeme.csv is prompted

Step 6

if need to redo a step or add more data just remove
the assembled_changeme.csv file and repeat step 5


Plotting and Statistics
=======================

This section describe the true power of phenolinks

