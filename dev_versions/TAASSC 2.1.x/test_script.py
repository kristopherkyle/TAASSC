import TAASSC_213_dev as lgr
import glob
### Sample Data analyses here
try1 = lgr.LGR_Analysis("They said she liked hamburgers. They also said that he didn't.")
try1["lemma_text"] #simple pos-specific lemmatized text
try1["nn_all"] #count for "nn_all" category (all nouns)
try1["tagged_text"] #all tags
lgr.print_vertical(try1["tagged_text"]) #pretty-print tags
lgr.output_vertical(try1["tagged_text"],"try1.tsv",ordered_output = "full") #write vertical output to file
lgr.output_xml(try1["tagged_text"],"try1.xml") #write xml output to file
file_list = glob.glob("test_files/*.txt") #create your own filelist
lgr.LGR_Full(file_list,"results.csv")
lgr.LGR_Full("test_files/","results2.csv") #let TAASSC generate the filelist based on a folder name
lgr.LGR_Full("test_files/","results2.csv",output = ["xml"]) #generate summary count file, generate xml representation for each
lgr.LGR_Full("test_files/","results2.csv",output = ["xml","vertical"]) #generate summary count file, generate xml representation  and vertical representation for each



##############################################################################################################
### need to add a function for counting tags from fix-tagged files [xml first, then possibly vert as well] ###
##############################################################################################################

#then, add try:except statements for Spacy

#Done:
#next, check scripts for generating output for a set of texts