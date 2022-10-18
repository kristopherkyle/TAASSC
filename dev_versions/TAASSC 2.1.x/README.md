# TAASSC Version 2.1.x

This folder represents in-development versions of TAASSC 2.1.

The code presumes you have installed Spacy (tested with Spacy 3.2) and have downloaded the "en_core_web_trf" model.

See test_script.py for example usage (also see below).

## import the package

```python
#until this package is on PyPI, make sure that your working directory is the folder that includes TAASSC_215_dev.py
import TAASSC_215_dev as lgr
```

## Process a string
```python
try1 = lgr.LGR_Analysis("They said she liked hamburgers. They also said that he didn't.")
print(try1["lemma_text"]) #simple pos-specific lemmatized text
print(try1["nn_all"]) #count for "nn_all" category (all nouns)
print(try1["tagged_text"]) #all tags
```
## Other output types
```python
lgr.print_vertical(try1["tagged_text"]) #pretty-print tags
lgr.output_vertical(try1["tagged_text"],"try1.tsv",ordered_output = "full") #write vertical output to file
lgr.output_xml(try1["tagged_text"],"try1.xml") #write xml output to file
```

## Process all files in a particular folder
``` python
lgr.LGR_Full("test_files/","results2.csv") #summary output only (spreadsheet); let TAASSC generate the filelist based on a folder name
lgr.LGR_Full("test_files/","results2.csv",output = ["xml"]) #generate summary count file (spreadsheet), generate xml representation for each
lgr.LGR_Full("test_files/","results2.csv",output = ["xml","vertical"]) #generate summary count file (spreadsheet), generate xml representation  and vertical representation for each
```

## Process all files in a list of filenames

```python
file_list = glob.glob("test_files/*.txt") #create your own filelist
lgr.LGR_Full(file_list,"results.csv")
```
## Recalculate indices from a folder of fix-tagged xml files

```python
XmlFileList = glob.glob('xml_output/*.xml') #list of files
lgr.lgrXml(XmlFileList,"xml_test.csv")
```
## To do:
- add more functionality to functions that read fix-tagged xml files
- further evaluate annotation accuracy, make tweaks
- add exception handling
- create python package (add relative links to source files, upload to pypi)
- more documentation (including feature descriptions)



