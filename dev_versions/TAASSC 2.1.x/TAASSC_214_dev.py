#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Tags in English tagset not in Spanish tagset:
# {'acomp', 'preconj', 'pcomp', 'intj', 'attr', 'agent', 'auxpass', 'poss', 'prep', 'pobj', 'dative', 'prt', 'expl', 'npadvmod', 'predet', 'quantmod', 'nsubjpass', 'oprd', 'meta', 'relcl', 'csubjpass', 'dobj', 'neg'}
# preconj is not referenced in TAASC

"""
Created on Fri Apr 10 11:47:46 2020

@author: kkyle2

This script builds on TAASSC 2.0.0.58, which was used in Kyle et al. 2021a,b.
The data for those publications were processed using 
- TAASSC 2.0.0.58
- Python version 3.7.3
- spaCy version 2.1.8
- spaCy `en_core_web_sm` model version 2.1.0.

This version of the code is part of a project designed to:
a) make a more user-friendly interface (including a python package, and online tool, and a desktop tool)
b) ensure that the tags are as accurate as possible

License:
The Tool for the Automatic Analysis of Syntactic Sophistication and Complexity (text analysis program)
    Copyright (C) 2020  Kristopher Kyle

This program is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

See https://creativecommons.org/licenses/by-nc-sa/4.0/ for a summary of the license (and a link to the full license).

"""
### imports ####################################
version = "2.1.14"
version_notes = "2.1.14 adds the ability to recalculate feature counts from edited xml files"

from calendar import c
import glob #for finding all filenames in a folder
import os #for making folders
from xml.dom import minidom #for pretty printing
import xml.etree.ElementTree as ET #for xml parsing
from random import sample #for random samples
import re

from sklearn.metrics import multilabel_confusion_matrix #for regulat expressions
#from lexical_diversity import lex_div as ld #for lexical diversity. Should probably upgrade to TAALED
import itertools 
from collections import defaultdict

### spacy
import spacy #base NLP
#nlp = spacy.load("en_core_web_sm") #load model


SPACY_MODEL = "en_core_web_trf"
nlp = spacy.load(SPACY_MODEL)  #load model
nlp.max_length = 1728483 

# list of dependency tags for spacy
DEPENDENCY_TAG_SET = nlp.get_pipe("parser").labels
# these are the coarse grained POS SpaCy tags. They are language agnostic.
POS_TAG_SET ={
	'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 
	'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE' 
}
# these are the fine grained tag SpaCy tags
TAG_SET = nlp.get_pipe("tagger").labels

SEMANTIC_NOUN_FILE = "lists_LGR/semantic_class_noun.txt"
SEMANTIC_VERB_FILE = "lists_LGR/semantic_class_verb.txt"
SEMANTIC_ADJECTIVE_FILE = "lists_LGR/semantic_class_adj.txt"
SEMANTIC_ADVERB_FILE = "lists_LGR/semantic_class_adverb_5-25-20.txt"
NOMINAL_STOP_FILE = "lists_LGR/nom_stop_list_edited.txt"
INDEX_LIST_FILE = "lists_LGR/LGR_INDEX_LIST_5-26-20.txt"
PERSONAL_PRONOUN1 = "i we our us my me ourselves myself".split(" ")
PERSONAL_PRONOUN2 = "you your yourself ya thy thee thine".split(" ")
PERSONAL_PRONOUN3 = "he she they their his them her him themselves himself herself".split(" ") #no "it" following Biber et al., 2004
PERSONAL_PRONOUN_IT = ["it"]

INDEFINITE_LIST = "everybody everyone everything somebody someone something anybody anyone anything nobody noone none nothing one ones".split(" ")
# note that "one" captures "no one"
# indefinite list from Longman grammar pp. 352-355
DEMONSTRATIVE_LIST = ["this","that","these","those"]

CONTRACTIONS_LIST = "'m 'll n't 're 's".split(" ")

PROPER_NOUN_NOMINALIZATION = ["an","ian","ans","ians"]

NOUN_NOMINALIZATION = [
		"al","cy","ee","er","or","ry",
		"ant","ent","dom","ing","ity","ure","age","ese","ess","ful","ism","ist",
		"ite","let","als","ees","ers","ors","ate","ance","ence","ment","ness","tion",
		"ship","ette","hood","cies","ries","ants","ents","doms","ings","ages","fuls",
		"isms","ists","ites","lets","eses","ates","ician","ities","ances","ences",
		"ments","tions","ships","esses","ettes","hoods","nesses"
]

NOUN_TAGS = ["NOUN", "PROPN"]
assert POS_TAG_SET.issuperset(NOUN_TAGS)
NONWORD_TAGS = ["PUNCT","SYM","SPACE","X"]
assert POS_TAG_SET.issuperset(NONWORD_TAGS)


THAT0_LIST = "check consider ensure illustrate fear say assume understand hold appreciate insist feel reveal indicate wish decide express follow suggest saw direct pray observe record imagine see think show confirm ask meant acknowledge recognize need accept contend come maintain believe claim verify demonstrate learn hope thought reflect deduce prove find deny wrote read repeat remember admit adds advise compute reach trust yield state describe realize expect mean report know stress note told held explain hear gather establish suppose found use fancy submit doubt felt".split(" ")
TO_VERB_LIST = "to_speech_act_verb cognition_verb desire_verb to_causative_verb probability_verb".split(" ")
TO_ADJECTIVE_LIST = "certainty_adj ability_willingness_adj personal_affect_adj ease_difficulty_adj evaluative_adj".split(" ")

MODAL_POSSIBILITY_LIST = "can may might could".split(" ")
MODAL_NECESSITY_LIST = "ought must should".split(" ")
MODAL_PREDICTIVE_LIST = "will would shall".split(" ")	

BE_LEMMA = ["be"] # In Spanish be has two variants SER and ESTAR

SEMANTIC_NOUN_CATEGORIES = "nn_animate nn_cognitive nn_concrete nn_technical nn_quantity nn_place nn_group nn_abstract".split(" ")

SEMANTIC_VERB_CATEGORIES = "activity_verb communication_verb mental_verb causation_verb occurrence_verb existence_verb aspectual_verb that_nonfactive_verb attitudinal_verb factive_verb likelihood_verb".split(" ")
INTRANSITIVE_PHRASAL_CATEGORIES = "intransitive_activity_phrasal_verb intransitive_occurence_phrasal_verb copular_phrasal_verb intransitive_aspectual_phrasal_verb".split(" ")
TRANSITIVE_PHRASAL_CATEGORIES = "transitive_activity_phrasal_verb transitive_mental_phrasal_verb transitive_communication_phrasal_verb".split(" ")

SEMANTIC_ADJECTIVE_CATEGORIES = "size_attributive_adj time_attributive_adj color_attributive_adj evaluative_attributive_adj relational_attributive_adj topical__attributive_adj".split(" ")

ADVERB_TYPE_LIST = "discourse_particle place_adverbials time_adverbials conjuncts_adverb downtoners_adverb hedges_adverb amplifiers_adverb emphatics".split(" ")
SEMANTIC_ADVERB_CATEGORIES = "attitudinal_adverb factive_adverb likelihood_adverb nonfactive_adverb".split(" ")


NOUN_PHRASE_DEPENDENCY_TAGS = ["relcl","amod","det","prep","poss","cc"]
assert DEPENDENCY_TAG_SET.issuperset(NOUN_PHRASE_DEPENDENCY_TAGS)

SEMANTIC_THAT_VERB_CATEGORIES = "nonfactive_verb attitudinal_verb factive_verb likelihood_verb".split(" ")
SEMNATIC_THAT_NOUN_CATEGORIES = "nn_nonfactive nn_attitudinal nn_factive_noun nn_likelihood"



# These are auxilliary lists that are definied automatically from the 
# NOUN_NOMINALIZATION and PROPER_NOUN_NOMINALIZATION lists

NOUN_NOMINALIZATION_BY_LENGTH = defaultdict(list)
for nn in NOUN_NOMINALIZATION:
	NOUN_NOMINALIZATION_BY_LENGTH[len(nn)].append(nn)
NOUN_NOMINALIZATION_MIN = min(NOUN_NOMINALIZATION_BY_LENGTH.keys())
NOUN_NOMINALIZATION_MAX = max(NOUN_NOMINALIZATION_BY_LENGTH.keys())

PROPER_NOUN_NOMINALIZATION_BY_LENGTH = defaultdict(list)
for pnn in PROPER_NOUN_NOMINALIZATION:
	PROPER_NOUN_NOMINALIZATION_BY_LENGTH[len(pnn)].append(pnn)
PROPER_NOUN_NOMINALIZATION_MIN = min(PROPER_NOUN_NOMINALIZATION_BY_LENGTH.keys())
PROPER_NOUN_NOMINALIZATION_MAX = max(PROPER_NOUN_NOMINALIZATION_BY_LENGTH.keys())




#allow more characters to be processed than default. This allows longer documents to be processed. This may need to be made longer.
######################################################

### Load lists, etc. #################################
### Define word-based lists ### (note, need to add phrasal items, also need to cross-reference with B 2006):

### Change both the way that the files are processed and the file format. 

def read_category_file(file_path):
	""" 
	Reads a file with lines that begin with a category name and then a tab separated list
	of words or phrases. The file is allowed to contain comments that begin with #. In order to
	accommodate multiple collections of categories in one file, a comment that begins with
	#### begins a new collection. If there is a single collection in a file, a single dictionary 
	is returned. If there are multiple collections in a file a list of dictionaries is returned.  

	Args:
		file_path (string): path to the category file.  
	"""
	with open(file_path,'r') as category_file:
		multiple_dictionaries = []
		curr_dict = {}
		for line in category_file:
			line = line.rstrip("\n")
			if line.startswith("####") and curr_dict != {}:
				multiple_dictionaries.append(curr_dict)
				curr_dict = {}
			elif not line.startswith("#"):
				category , *words = line.split("\t")
				curr_dict.update({word : category for word in words})
		if curr_dict != {}:
			multiple_dictionaries.append(curr_dict)
		return (
			multiple_dictionaries[0] if len(multiple_dictionaries) == 1 
			else multiple_dictionaries
		)

NOUN_DICT = read_category_file(SEMANTIC_NOUN_FILE)
VERB_DICT, THAT_VERB_DICT, TO_VERB_DICT, PHRASAL_VERB_DICT = read_category_file(SEMANTIC_VERB_FILE)
ADJECTIVE_DICT = read_category_file(SEMANTIC_ADJECTIVE_FILE)
ADVERB_DICT = read_category_file(SEMANTIC_ADVERB_FILE) 
NOMINAL_STOP = open(NOMINAL_STOP_FILE).read().split("\n")


""" def list_dict(tabsep_lines):
	return { word : category for line in tabsep_lines for category, *words in line.split("\t") for word in words}


semantic_noun = open(SEMANTIC_NOUN_FILE).read().split("\n")
semantic_verb = open(SEMANTIC_VERB_FILE).read().split("\n")
semantic_adj = open(SEMANTIC_ADJECTIVE_FILE).read().split("\n")
semantic_adv = open(SEMANTIC_ADVERB_FILE).read().split("\n")
nominal_stop = open(NOMINAL_STOP_FILE).read().split("\n")

# Why the slices. Intend to replace this with a new word list file format that allows the splits to be directly
# specified in the word files.
NOUN_DICT = list_dict(semantic_noun)
VERB_DICT = list_dict(semantic_verb[:7])
THAT_VERB_DICT = list_dict(semantic_verb[7:11])
TO_VERB_DICT = list_dict(semantic_verb[11:16]) # Is this correct?
PHRASAL_VERB_DICT = list_dict(semantic_verb[16:])
ADJECTIVE_DICT = list_dict(semantic_adj)
ADVERB_DICT = list_dict(semantic_adv) """

### Define categories and indices ########
##########################################
cat_to_tag = {
	"main_tag": "nn_all prep_phrase verb pp_all wh_relative_clause".split(" "), #create list of possible main tags
	"main_tag2": ["all_phrasal_verbs"], #create list of possible main tags
	"spec_tag1": "nominalization pp1 pp2 pp3 pp3_it pp_indefinite pp_demonstrative cc_phrase cc_clause wh_question past_tense perfect_aspect non_past_tense jj_attributive jj_predicative discourse_particle place_adverbials time_adverbials conjuncts_adverb downtoners_adverb hedges_adverb amplifiers_adverb emphatics wh_clause wh_relative_subj_clause wh_relative_obj_clause wh_relative_prep_clause that_relative_clause that_complement_clause".split(" "), #create list of possible spec_tag1 tags
	"spec_tag2": "pv_do split_aux be_mv that_verb_clause that_adjective_clause that_noun_clause".split(" "),
	"spec_tag3": "adverbial_subordinator_causitive adverbial_subordinator_conditional adverbial_subordinator_other agentless_passive by_passive".split(" "),
	"spec_tag4": ["contraction","to_clause"],
	"spec_tag5": "modal_possibility modal_necessity modal_predictive to_clause_noun to_clause_verb to_clause_adjective".split(" "),
	"spec_tag6": ["past_participial_clause", "complementizer_that0"],
	"semantic_tag1": "nn_animate nn_cognitive nn_concrete nn_technical nn_quantity nn_place nn_group nn_abstract activity_verb communication_verb mental_verb causation_verb occurrence_verb existence_verb aspectual_verb intransitive_activity_phrasal_verb intransitive_occurence_phrasal_verb copular_phrasal_verb intransitive_aspectual_phrasal_verb transitive_activity_phrasal_verb transitive_mental_phrasal_verb transitive_communication_phrasal_verb size_attributive_adj time_attributive_adj color_attributive_adj evaluative_attributive_adj relational_attributive_adj topical__attributive_adj attitudinal_adj likelihood_adj certainty_adj ability_willingness_adj personal_affect_adj ease_difficulty_adj evaluative_adj attitudinal_adverb factive_adverb likelihood_adverb nonfactive_adverb that_verb_clause_nonfactive that_verb_clause_attitudinal that_verb_clause_factive that_verb_clause_likelihood that_noun_clause_nonfactive that_noun_clause_attitudinal that_noun_clause_factive that_noun_clause_likelihood to_adjective_clause_certainty to_adjective_clause_ability_willingness to_adjective_clause_personal_affect to_adjective_clause_ease_difficulty to_adjective_clause_evaluative that_adjective_clause_attitudinal that_adjective_clause_likelihood".split(" "),	
	"semantic_tag2": "to_clause_verb_to_speech_act to_clause_verb_cognition to_clause_verb_desire to_clause_verb_to_causative to_clause_verb_probability to_clause_adjective_certainty to_clause_adjective_ability_willingness to_clause_adjective_personal_affect to_clause_adjective_ease_difficulty to_clause_adjective_evaluative".split(" "),
	"other": "wrd_length nwords mattr".split(" "),
}

# This checks that the categories defined in cat_to_tag are disjoint
cat_pairs = list(itertools.combinations(cat_to_tag.keys(),2))
assert all(set(cat_to_tag[cat1]).isdisjoint(cat_to_tag[cat2]) for cat1, cat2 in list(cat_pairs))

cat_dict = { tag : cat for cat in cat_to_tag for tag in cat_to_tag[cat]}

cats = {} #here we use a dictionary instead of a list for speed.
for x in "main_tag spec_tag1 spec_tag2 spec_tag3 spec_tag4 spec_tag5 spec_tag6 semantic_tag1 semantic_tag2".split(" "):
	cats[x] = None

INDEX_LIST = open(INDEX_LIST_FILE).read().split("\n")

##########################################
### Utility Functions ####################

def ex_tester(input_text): #example tester (for checking Spacy output)
	spcy_sample = nlp(input_text)
	sent_number = 1
	for sent in spcy_sample.sents:
		print("sent_number" + str(sent_number))
		sent_number +=1
		for token in sent:
			print(token.text, token.lemma_, token.tag_, token.pos_, token.dep_, token.head.text, token.i)

def safe_divide(numerator,denominator):
	if float(denominator) == 0.0:
		return(0.0)
	else:
		return(numerator/denominator)
		
###########################################
#### other functions ###

def prettify(elem):
	"""Return a pretty-printed XML string for the Element.
	"""
	rough_string = ET.tostring(elem, 'utf-8')
	reparsed = minidom.parseString(rough_string)
	return(reparsed.toprettyxml(indent="    "))


def wrd_nchar(token,feature_dict): #following B et al 2004	
	if token.pos_ not in NONWORD_TAGS:
		feature_dict["wrd_length"] += len(token.text)
		feature_dict["nwords"] += 1
		if token.lemma_ == "-PRON-":
			lemma = token.text.lower()
		else:
			lemma = token.lemma_
		feature_dict["lemma_text"].append(f"{lemma}_{token.pos_}")


def noun_phrase_complexity(token,feature_dict):
	if token.pos_ == "NOUN": #only consider common nouns (exclude pronouns and proper nouns)
		feature_dict["np"] += 1
		deps = [child.dep_ for child in token.children]
		feature_dict["np_deps"] += len(deps)
		for x in deps: # record complexity of noun-phrases
			if x in NOUN_PHRASE_DEPENDENCY_TAGS:
				feature_dict["{x}_dep"] += 1
			
def clausal_complexity(token,feature_dict):
	if token.pos_ == "VERB":
		if token.dep_ != "aux": #check to make sure that this is a main verb
			feature_dict["all_clauses"] += 1 #finite and nonfinite clauses. Probably won't use
			deps = [child.dep_ for child in token.children]
			if "nsubj" in deps or "nsubjpass" in deps:
				feature_dict["finite_clause"] += 1
				if token.dep_ in ["ROOT","conj"]:
					feature_dict["finite_ind_clause"] += 1
				else:
					feature_dict["finite_dep_clause"] += 1
				if token.dep_ == "ccomp":
					feature_dict["finite_compl_clause"] += 1
				if token.dep_ == "relcl":
					feature_dict["finite_relative_clause"] += 1
			else:
				feature_dict["nonfinite_clause"] += 1 #can use "to_clause" to distinguis
			feature_dict["vp_deps"] += len(deps)

###########################################
def basic_info(token, token_d):
	token_d["word"] = token.text
	token_d["lemma"] = token.lemma_.lower()
	token_d["pos"] = token.pos_
	token_d["tag"] = token.tag_
	token_d["idx"] = str(token.i)
	token_d["dep_rel"] = token.dep_
	token_d["head"] = token.head.text
	token_d["head idx"] = str(token.head.i)
	
### Linguistic Analysis Functions ###
def pronoun_analysis(token,token_d,feature_dict): #takes spaCY token object and feature_dict as arguments
	token_lower = token.text.lower()
	#NOTE: Spacy tags "our" and "my" as determiners - run with out tags
	pp_all = (PERSONAL_PRONOUN1 +PERSONAL_PRONOUN2
					+PERSONAL_PRONOUN3+PERSONAL_PRONOUN_IT)
	if token_lower in pp_all:
		feature_dict["pp_all"] += 1 #add one to feature dict (this is external to the function)
		token_d["main_tag"] = "pp_all"
	if token_lower in PERSONAL_PRONOUN1:
		feature_dict["pp1"] += 1 #add one to feature dict (this is external to the function)
		token_d["spec_tag1"] = "pp1"
	elif token_lower in PERSONAL_PRONOUN2:
		feature_dict["pp2"] += 1
		token_d["spec_tag1"] = "pp2"
	elif token_lower in PERSONAL_PRONOUN3:
		feature_dict["pp3"] += 1
		token_d["spec_tag1"] = "pp3"
	elif token_lower in PERSONAL_PRONOUN_IT:
		feature_dict["pp3_it"] += 1
		token_d["spec_tag1"] = "pp3_it"

def advanced_pronoun(token,doc,token_d,feature_dict):	#updated 5-8-2020
	token_lower = token.text.lower()	
	if token_lower in INDEFINITE_LIST and token.dep_ in ["nsubj","nsubjpass","dobj","pobj"]:
		feature_dict["pp_indefinite"] += 1
		token_d["spec_tag1"] = "pp_indefinite"
	elif token_lower in DEMONSTRATIVE_LIST: 
		if token.dep_ == "advmod":
			feature_dict["pp_demonstrative"] += 1
			token_d["spec_tag1"] = "pp_demonstrative"
		elif token.i + 1 < len(doc) and doc[token.i + 1].text.lower() in ["who",".","!","?",":"]:
			feature_dict["pp_demonstrative"] += 1
			token_d["spec_tag1"] = "pp_demonstrative"		
		elif token.dep_ == "nsubjpass":
			if token.head.dep_ != "relcl":
				feature_dict["pp_demonstrative"] += 1
				token_d["spec_tag1"] = "pp_demonstrative"
		elif token.dep_ == "pobj":
				feature_dict["pp_demonstrative"] += 1
				token_d["spec_tag1"] = "pp_demonstrative"				
		elif token.dep_ in ["nsubj","dobj"]:
			if token.head.dep_ != "relcl" and token.head.pos_ != "NOUN":
				feature_dict["pp_demonstrative"] += 1
				token_d["spec_tag1"] = "pp_demonstrative"

def pro_verb(token,token_d,feature_dict):
	transitive = False
	
	if token.lemma_ == "do" and token.pos_ == "VERB" and token.dep_ != "aux":
		for x in token.children:
			if x.dep_ in ["dobj","ccomp"]: #if do is transitive:
				transitive = True
		if transitive == False: #if it isn't transitive, then it is a pro-verb
			feature_dict["pv_do"] += 1
			token_d["spec_tag2"] = "pv_do"

def contraction_check(token,token_d,feature_dict):	
	if token.text.lower() in CONTRACTIONS_LIST and token.dep_ != "case":
		feature_dict["contraction"] += 1
		token_d["spec_tag4"] = "contraction"

def split_aux_check(token,token_d,feature_dict): 
	#takes a verb as input and checks for adverbs between auxilliary and main verb
	if token.pos_ == "VERB" and token.dep_ not in ["aux", "aux_pass"]:
		end = token.i #verb position
		start = False #for first aux position
		adv = False #for adverb
		
		for x in token.children:
			if x.dep_ in ["aux", "aux_pass"]:
				if start == False:
					start = x.i
				elif start != False and x.i < start:
					start = x.i
			elif x.dep_ == "advmod":
				adv = x.i
		if start != False and adv != False:
			if adv > start and adv < end:
				feature_dict["split_aux"] += 1
				token_d["spec_tag2"] = "split_aux"
	
def prep_analysis(token,token_d,feature_dict):	
	# Other possible adverbial subordinators are although, when, while.
	if token.dep_ == "mark":
		if token.text.lower() == "because": # NOTE: this list is likely incomplete
			feature_dict["adverbial_subordinator_causitive"] +=1
			token_d["spec_tag3"] = "adverbial_subordinator_causitive"
		
		elif token.text.lower() in ["if", "unless"]: # # NOTE:this list is likely incomplete
			feature_dict["adverbial_subordinator_conditional"] +=1
			token_d["spec_tag3"] = "adverbial_subordinator_conditional"
			
		elif token.text.lower() not in ["that"]:
			feature_dict["adverbial_subordinator_other"] +=1
			token_d["spec_tag3"] = "adverbial_subordinator_other"
	
	elif token.dep_ == "prep":
		obj = False
		for x in token.children:
			# pcomp is not present in the UD tagset for Spanish in SpaCy
			if x.dep_ in ["pobj", "pcomp", "prep","amod", "cc"]:
				obj = True
			if x.dep_ == "punct" and x.text not in ". , ? !": #this deals with some weirdness in spacy where some numbers and mathematical symbols (e.g., "x" and "y") were not counted as prepositional objects.
				obj = True
		#Updated 5-25-20
		if obj == True: #check to make sure that there is a prepositional object or another prepositional phrase
			#attribute = "prep_phrase"
			feature_dict["prep_phrase"] += 1
			token_d["main_tag"] = "prep_phrase"
		

#this may need to be updated
def coordination_analysis(token,wrd_count,token_d,feature_dict):
		if token.text.lower() in ["and", "or"]: #only consider and/or
			if wrd_count == 0:
				feature_dict["cc_clause"] += 1
				token_d["spec_tag1"] = "cc_clause"
				
			elif token.head.pos_ in ["NOUN", "ADJ", "ADV", "PRON", "PROPN", "PART"]:
				feature_dict["cc_phrase"] += 1
				token_d["spec_tag1"] = "cc_phrase"
			
			elif token.head.pos_ == "VERB": 
				l = [] #list for verbs and ccs	
				for child in token.head.children: #iterate the children of the main verb (1st verb)
						#print(child.text, child.i, child.dep_) #for debugging
						if child.dep_ == "cc":
							l.append([child.i,"cc"]) #add cc index to list
						if child.dep_ == "conj": #narrow down the target of analysis to the second verb(s) by specifying the analysis to corrdinated second element
							
							if "nsubj" in [chld.dep_ for chld in child.children]: #identify if the second element has "nsubj" tag in its children, if so, this is independent clause coordination
								l.append([child.i,"cc_clause"]) #add index and coordination type to list
							elif "nsubj" not in [chld.dep_ for chld in child.children]: # if not it is phrasal
								l.append([child.i,"cc_phrase"]) #add index and coordination type to list
				
				sort_list = sorted(l,key=lambda x: x[0]) #sort list by index to put in order of appearance
				for x in range(len(sort_list)): #iterate through list of numbers that is the same length as list
					if sort_list[x][0] == token.i: #if the list item has the same index as our target token:
						try:
							relation = sort_list[x+1][1]
							if relation == "cc": #edited 4-25, my need to be changed
								relation = "cc_clause"
							feature_dict[relation] +=1 #add one to the count for the next token in the list (i.e., a verb attribute)
							token_d["spec_tag1"] = relation #set attribute as next token's attribute (this will be a verb)
							break
						except IndexError:
							continue

def noun_analysis(token,token_d,feature_dict): #revised 5/8/20.
	if token.pos_ in NOUN_TAGS:

		feature_dict["nn_all"] += 1 #add one to the noun count
		token_d["main_tag"] = "nn_all" #add main tag to attributes

		token_lower = token.text.lower() 

		#note that zero derivation is not included
		#stop list comes from list derived from tagged T2KSWAL and hand edited
		if token_lower not in NOMINAL_STOP:
			if token.pos_ == "PROPN":
				for l in range(min(len(token_lower)-1, PROPER_NOUN_NOMINALIZATION_MAX), PROPER_NOUN_NOMINALIZATION_MIN-1,-1):
					if token_lower[-l:] in PROPER_NOUN_NOMINALIZATION_BY_LENGTH[l]:
						feature_dict["nominalization"] += 1
						token_d["spec_tag1"] = "nominalization"
						break
			for l in range(min(len(token_lower)-1, NOUN_NOMINALIZATION_MAX), NOUN_NOMINALIZATION_MIN-1,-1):
				if token_lower[-l:] in PROPER_NOUN_NOMINALIZATION_BY_LENGTH[l]:
					feature_dict["nominalization"] += 1
					token_d["spec_tag1"] = "nominalization"
					break

def semantic_analysis_noun(token, token_d,feature_dict):
	if token.pos_ in NOUN_TAGS:
		lemma = token.lemma_.lower() #set lemma form of word
		if lemma in NOUN_DICT and NOUN_DICT[lemma] in SEMANTIC_NOUN_CATEGORIES: #check that word is in dict and category is in var_list
			feature_dict[NOUN_DICT[lemma]] +=1 #if so, add one to feature_dict
			token_d["semantic_tag1"] = NOUN_DICT[lemma] #set attribute
			

def be_analysis(token,token_d,feature_dict):
	if token.lemma_.lower() in BE_LEMMA and token.dep_ not in ["aux","auxpass"]:
		feature_dict["be_mv"] += 1
		token_d["spec_tag2"] = "be_mv"
	

def verb_analysis(token,doc_text,token_d,feature_dict): #need to add spearate tags for tense/aspect and passives


	if token.pos_ == "VERB":
		feature_dict["verb"] += 1 #add one to verb count
		token_d["main_tag"] = "verb"
		
		if token.dep_ == "aux": #check for auxilliaries
			
			if token.text in MODAL_POSSIBILITY_LIST: # if in modal list
				feature_dict["modal_possibility"] += 1
				token_d["spec_tag5"] = "modal_possibility"
			
			elif token.text in MODAL_NECESSITY_LIST: # if in modal list
				feature_dict["modal_necessity"] += 1
				token_d["spec_tag5"] = "modal_necessity"
		
			elif token.text in MODAL_PREDICTIVE_LIST: # if in modal list
				feature_dict["modal_predictive"] += 1
				token_d["spec_tag5"] = "modal_predictive"
			
			elif token.tag_ == "VBD": #otherwise, count it as past tense
				feature_dict["past_tense"] += 1 #add one to count
				token_d["spec_tag4"] = "past_tense"
			
			
			else:
				feature_dict["non_past_tense"] += 1 #add one to count
				token_d["spec_tag1"] = "non_past_tense"
		
		else: # if not an auxilliary
			#need to exclude infinitives and reflexive pronouns
			if token.head.lemma_ in THAT0_LIST and token.dep_ == "ccomp" and token.i > token.head.i:
				that0_problem = False
				finite = False
				aux_be = False
				vbg_problem = False
				for x in token.children:
					if x.text.lower() in ["that","who","what","how","where","why","when","whose","whom","whomever"] and x.dep_ != "det": #this will likely be overly disriminitory
						that0_problem = True
					if x.dep_ == "mark":
						that0_problem = True	
					if x.dep_ in ["nsubj","csubj"]:
						finite = True
					if x.dep_ == "aux" and x.lemma_ == "be":
						aux_be = True
				if token.tag_ == "VBG" and aux_be == False:
					vbg_problem = True
				if that0_problem == False and doc_text[token.head.i - 1].text not in ["that","who","what","how","where","why","when","whose","whom","whomever","whatever","which"]:
					if doc_text[token.head.i + 1].text not in ["that","who","what","how","where","why","when","whose","whom","whomever","whatever","which",'"',"'",",",":","myself","itself","herself","ourself","ourselves","themselves","themself"]:
						if " ".join([doc_text[token.head.i + 1].text,doc_text[token.head.i + 2].text]) not in ["' ,",'" ,'] and "dobj" not in [child.dep_ for child in token.head.children]:
							if finite == True and vbg_problem == False:
								feature_dict["complementizer_that0"] += 1 #add one to count
								token_d["spec_tag6"] = "complementizer_that0"

			if token.dep_ == "acl" and token.tag_ == "VBN":
				feature_dict["past_participial_clause"] += 1 #add one to count
				token_d["spec_tag6"] = "past_participial_clause"
			
			if doc_text[token.i-1].text.lower() == "to" and doc_text[token.i-1].dep_ == "aux" and doc_text[token.i-1].head.i == token.i:
				contr_token = doc_text[token.i-2]
				if contr_token.text.lower() not in ["able","ought"]: #spacy wasn't getting all of the phrasal verbs (that tag "to" as "part" instead of "aux") More may need to be added here
					feature_dict["to_clause"] += 1 #add one to count
					token_d["spec_tag4"] = "to_clause"
					
					if contr_token.pos_ == "NOUN": #consider using head.text instead (previous version used that)
						feature_dict["to_clause_noun"] += 1 #add one to count
						token_d["spec_tag5"] = "to_clause_noun"
					
					if contr_token.pos_ == "VERB":
						feature_dict["to_clause_verb"] += 1 #add one to count
						token_d["spec_tag5"] = "to_clause_verb"

						if contr_token.lemma_ in TO_VERB_DICT and TO_VERB_DICT[contr_token.lemma_] in TO_VERB_LIST:
							feature_dict["to_clause_verb_" + TO_VERB_DICT[contr_token.lemma_][:-5]] += 1
							token_d["semantic_tag2"] = "to_clause_verb_" + TO_VERB_DICT[contr_token.lemma_][:-5]

					if doc_text[token.i-2].pos_ == "ADJ":
						feature_dict["to_clause_adjective"] += 1 #add one to count
						token_d["spec_tag5"] = "to_clause_adjective"
						
						if contr_token.lemma_ in ADJECTIVE_DICT and ADJECTIVE_DICT[contr_token.lemma_] in TO_ADJECTIVE_LIST:
							feature_dict["to_clause_adjective_" + ADJECTIVE_DICT[contr_token.lemma_][:-4]] += 1
							token_d["semantic_tag2"] = "to_clause_adjective_" + ADJECTIVE_DICT[contr_token.lemma_][:-4]

			if token.tag_ == "VBD":
				feature_dict["past_tense"] += 1 #add one to count
				token_d["spec_tag1"] = "past_tense"
			
			elif token.tag_ in ["VBN","VBG"]:
				for x in token.children:
					if x.lemma_ == "have" and x.dep_ == "aux":
						feature_dict["perfect_aspect"] += 1 #add one to count
						token_d["spec_tag1"] = "perfect_aspect"
						break
			else:
				feature_dict["non_past_tense"] += 1 #add one to count
				token_d["spec_tag1"] = "non_past_tense"

def	passive_analysis(token,token_d,feature_dict):
	if token.pos_ == "VERB":
		child_list = [] #list for child dependents		
		for x in token.children: #add dependents to the list - annoying, but necessary because of the way spaCy works
			child_list.append(x.dep_)
		
		if "auxpass" in child_list: #check for passives			
			if "agent" in child_list:
				feature_dict["by_passive"] += 1
				token_d["spec_tag3"] = "by_passive"
			else:
				feature_dict["agentless_passive"] += 1
				token_d["spec_tag3"] = "agentless_passive"

def semantic_analysis_verb(token,token_d,feature_dict):
	lemma = token.lemma_.lower() #set lemma form of word
	if token.pos_ == "VERB":	
		#distinguish between non-phrasal, intransitive phrasal and transitive phrasal verbs
		if "prt" in [chld.dep_ for chld in token.children]: # first, check for phrasal_verbs
			for x in token.children: #then extract particle text
				if x.dep_ == "prt":
					phrasal = lemma + " " + x.text #create phrasal verb - will only work with two-word phrasal verbs
					#print(phrasal)
					if phrasal in PHRASAL_VERB_DICT:
						feature_dict["all_phrasal_verbs"] +=1
						token_d["main_tag2"] = "all_phrasal_verbs"
					if "dobj" in [chld.dep_ for chld in token.children]: #check for transitivitivity
						if phrasal in PHRASAL_VERB_DICT and PHRASAL_VERB_DICT[phrasal] in TRANSITIVE_PHRASAL_CATEGORIES:
							feature_dict[PHRASAL_VERB_DICT[phrasal]] +=1 #if so, add one to feature_dict
							token_d["semantic_tag1"] = PHRASAL_VERB_DICT[phrasal] #set attribute
					else:
						if phrasal in PHRASAL_VERB_DICT and PHRASAL_VERB_DICT[phrasal] in INTRANSITIVE_PHRASAL_CATEGORIES:
							feature_dict[PHRASAL_VERB_DICT[phrasal]] +=1 #if so, add one to feature_dict
							token_d["semantic_tag1"] = PHRASAL_VERB_DICT[phrasal] #set attribute
		else:
			if lemma in VERB_DICT and VERB_DICT[lemma] in SEMANTIC_VERB_CATEGORIES: #check that word is in dict and category is in var_list
				feature_dict[VERB_DICT[lemma]] +=1 #if so, add one to feature_dict
				token_d["semantic_tag1"] = VERB_DICT[lemma] #set attribute

def adjective_analysis(token,token_d,feature_dict):
	#pred_list = "attitudinal_adj likelihood_adj certainty_adj ability_willingness_adj personal_affect_adj ease_difficulty_adj evaluative_adj".split(" ")
	# The acomp tag is not present in the UD tagset for Spanish in SpaCy
	if token.dep_ in ["acomp"]: #if dep relation is adjective complement, clausal complement, or object predicate (see Biber et al. 1999)
		feature_dict["jj_predicative"] += 1
		token_d["spec_tag1"] = "jj_predicative"
		if token.lemma_.lower() in ADJECTIVE_DICT and ADJECTIVE_DICT[token.lemma_.lower()] in SEMANTIC_ADJECTIVE_CATEGORIES:
			feature_dict[ADJECTIVE_DICT[token.lemma_.lower()]] += 1
			token_d["semantic_tag1"] = ADJECTIVE_DICT[token.lemma_.lower()]		
	elif token.dep_ == "amod":
		feature_dict["jj_attributive"] += 1
		token_d["spec_tag1"] = "jj_attributive"
		
def adverb_analysis(token, w_count, token_d,feature_dict):
	lemma = token.lemma_.lower()
	if token.pos_ == "ADV" or token.dep_ in ["npadvmod","advmod", "intj"]:
		if lemma in ADVERB_DICT and ADVERB_DICT[lemma] in ADVERB_TYPE_LIST:
			feature_dict[ADVERB_DICT[lemma]] += 1
			token_d["spec_tag1"] = ADVERB_DICT[lemma]	
		if w_count == 0 and token.text.lower() in "well now anyway anyhow anyways".split(" "):
			feature_dict["discourse_particle"] += 1
			token_d["spec_tag1"] = "discourse_particle"
		elif lemma in ADVERB_DICT and ADVERB_DICT[lemma] in SEMANTIC_ADVERB_CATEGORIES:
			feature_dict[ADVERB_DICT[lemma]] += 1
			token_d["semantic_tag1"] = ADVERB_DICT[lemma]

def wh_analysis(token,w_count,doc_text,sent_doc,token_d,feature_dict):	
	if token.tag_ in ["WDT","WP", "WP$", "WRB"] and token.text.lower() != "that":
		if token.head.dep_ not in ["csubj","ccomp", "pcomp"]:
			if w_count == 0 or doc_text[token.i-1].text in ['"',"'",":"]: #if WH word is first word in sentence or is first word in quote or after colon:		
				if "?" in [t.text for t in sent_doc]:
					feature_dict["wh_question"] += 1
					token_d["spec_tag1"] = "wh_question"
		if doc_text[token.i-1].pos_ == "VERB" and doc_text[token.i-1].lemma_ != "be":
			if token.head.dep_ != "advcl": #not sure if this is corret or not.
				feature_dict["wh_clause"] += 1
				token_d["spec_tag1"] = "wh_clause"

		if token.dep_ == "pobj" and token.head.head.dep_ == "relcl":
			feature_dict["wh_relative_clause"] += 1
			token_d["main_tag"] = "wh_relative_clause"			
			feature_dict["wh_relative_prep_clause"] += 1
			token_d["spec_tag1"] = "wh_relative_prep_clause"

		if token.head.dep_ == "relcl":
			if token.dep_ in ["nsubj","nsubjpass"]:
				feature_dict["wh_relative_clause"] += 1
				token_d["main_tag"] = "wh_relative_clause"			
				feature_dict["wh_relative_subj_clause"] += 1
				token_d["spec_tag1"] = "wh_relative_subj_clause"
			
			if token.dep_ in ["dobj"]:
				feature_dict["wh_relative_clause"] += 1
				token_d["main_tag"] = "wh_relative_clause"			
				feature_dict["wh_relative_obj_clause"] += 1
				token_d["spec_tag1"] = "wh_relative_obj_clause"

def that_analysis(token,doc_text,token_d,feature_dict):

	if token.text.lower() == "that":
		if token.dep_ in ["nsubj","nsubjpass","dobj","pobj"] and token.head.dep_ == "relcl": #consider adding "mark" to the possible token.dep_ options
			feature_dict["that_relative_clause"] += 1
			token_d["spec_tag1"] = "that_relative_clause"
		if token.dep_ in ["mark","nsubj"] and token.head.dep_ in ["ccomp","acl"]:
			feature_dict["that_complement_clause"] += 1
			token_d["spec_tag1"] = "that_complement_clause"	
			if doc_text[token.i-1].pos_ == "VERB":
				feature_dict["that_verb_clause"] += 1
				token_d["spec_tag2"] = "that_verb_clause"
				verb_lemma = doc_text[token.i-1].lemma_.lower()
				if verb_lemma in THAT_VERB_DICT and THAT_VERB_DICT[verb_lemma] in SEMANTIC_THAT_VERB_CATEGORIES: #check for semantic class
					feature_dict["that_verb_clause_" + THAT_VERB_DICT[verb_lemma][:-5]] += 1
					token_d["semantic_tag1"] = "that_verb_clause_" + THAT_VERB_DICT[verb_lemma][:-5]

			if doc_text[token.i-1].pos_ == "NOUN":
				feature_dict["that_noun_clause"] += 1
				token_d["spec_tag2"] = "that_noun_clause"
				noun_lemma = doc_text[token.i-1].lemma_.lower()
				#print(noun_lemma)
				if noun_lemma in NOUN_DICT and NOUN_DICT[noun_lemma] in SEMNATIC_THAT_NOUN_CATEGORIES:
					#print(noun_lemma,NOUN_DICT[noun_lemma])
					feature_dict["that_noun_clause_" + NOUN_DICT[noun_lemma][3:]] += 1
					token_d["semantic_tag1"] = "that_noun_clause_" + NOUN_DICT[noun_lemma][3:]


			if doc_text[token.i-1].pos_ == "ADJ":
				feature_dict["that_adjective_clause"] += 1
				token_d["spec_tag2"] = "that_adjective_clause"
				adj_lemma = doc_text[token.i-1].lemma_.lower()
				if adj_lemma in ADJECTIVE_DICT:
					if ADJECTIVE_DICT[adj_lemma] == "attitudinal_adj":
						feature_dict["that_adjective_clause_attitudinal"] +=1
						token_d["semantic_tag1"] = "that_adjective_clause_attitudinal"

					if ADJECTIVE_DICT[adj_lemma] == "likelihood_adj":
						feature_dict["that_adjective_clause_likelihood"] +=1
						token_d["semantic_tag1"] = "that_adjective_clause_likelihood"

#############################

#### These functions use the previous functions to conduct tagging and tallying of lexicogramamtical features ###

def LGR_Analysis(text,indices_dict=INDEX_LIST,cats_d = cats,output = False):
	index_dict = {}
	for x in indices_dict:
		index_dict[x] = 0 #start index counts
	index_dict["lemma_text"] = []
	def clean_text(in_text):
		
		if "[" in in_text and "]" in in_text:
			in_text = re.sub("\[.*?\]","",in_text)
		if "1:" in in_text:
			in_text = re.sub("\n[0-9]:","",in_text)
		if " " in in_text:
			if "\n" not in in_text:
				in_text = " ".join(in_text.split())
			else:
				line_text = []
				for x in in_text.split("\n"):
					line_text.append(" ".join(x.split()))
				in_text = "\n".join(line_text)
		return(in_text)
		
	doc = nlp(clean_text(text))
	
	output_list = []
	sent_idx = 0 #sentence counter
	for sent in doc.sents:
		output_list.append([])#add empty sentence-level list that will be filled below
		idx_sent = 0 #token witin sentence counter
		output_list.append([])
		for token in sent:
			#create token dictionary and populate it
			token_attrs = {}
			for x in cats_d:
				token_attrs[x] = None
			
			basic_info(token, token_attrs)
			pronoun_analysis(token,token_attrs,index_dict)
			advanced_pronoun(token,doc,token_attrs,index_dict)
			pro_verb(token,token_attrs,index_dict)
			
			contraction_check(token,token_attrs,index_dict)
			split_aux_check(token,token_attrs,index_dict)
			
			prep_analysis(token,token_attrs,index_dict)
			
			coordination_analysis(token,idx_sent,token_attrs,index_dict)
			
			wh_analysis(token,idx_sent,doc,sent,token_attrs,index_dict)
			
			noun_analysis(token,token_attrs,index_dict)
			semantic_analysis_noun(token,token_attrs,index_dict)
			
			be_analysis(token,token_attrs,index_dict)
			verb_analysis(token,doc,token_attrs,index_dict)
			passive_analysis(token,token_attrs,index_dict)
			semantic_analysis_verb(token,token_attrs,index_dict)
			
			adjective_analysis(token,token_attrs,index_dict)
			adverb_analysis(token,idx_sent,token_attrs,index_dict)
			
			that_analysis(token,doc,token_attrs,index_dict)
			wrd_nchar(token,index_dict)
			noun_phrase_complexity(token,index_dict)
			clausal_complexity(token,index_dict)

			output_list[sent_idx].append(token_attrs)
			idx_sent +=1
		sent_idx += 1
	
	
		
	index_dict["tagged_text"] = output_list
	index_dict["wrd_length"] = index_dict["wrd_length"]/index_dict["nwords"]
	index_dict["mattr"] = ld.mattr(index_dict["lemma_text"])
	#noun phrase complexity
	index_dict["mean_nominal_deps"] = safe_divide(index_dict["np_deps"],index_dict["np"]) #nominal (common noun) dependents,#of nominals (common nouns)
	index_dict["relcl_nominal"] = safe_divide(index_dict["relcl_dep"],index_dict["np"])
	index_dict["amod_nominal"] = safe_divide(index_dict["amod_dep"],index_dict["np"])
	index_dict["det_nominal"] = safe_divide(index_dict["det_dep"],index_dict["np"])
	index_dict["prep_nominal"] = safe_divide(index_dict["prep_dep"],index_dict["np"])
	index_dict["poss_nominal"] = safe_divide(index_dict["poss_dep"],index_dict["np"])
	index_dict["cc_nominal"] = safe_divide(index_dict["cc_dep"],index_dict["np"])

	index_dict["mean_verbal_deps"] = safe_divide(index_dict["vp_deps"],index_dict["finite_clause"]) #dependents per finite clause
	index_dict["mlc"] = safe_divide(index_dict["nwords"],index_dict["finite_clause"]) #number of words,number of finite clauses
	index_dict["mltu"] = safe_divide(index_dict["nwords"],index_dict["finite_ind_clause"]) #number of words,number of independent finite clauses (T-units)
	index_dict["dc_c"] = safe_divide(index_dict["finite_dep_clause"],index_dict["finite_clause"]) #number of dependent clauses,number of finite clauses (dependent clauses per clause)

	index_dict["ccomp_c"] = safe_divide(index_dict["finite_compl_clause"],index_dict["finite_clause"]) #complement clauses,number of finite clauses
	index_dict["relcl_c"] = safe_divide(index_dict["finite_relative_clause"],index_dict["finite_clause"]) #relative clauses,number of finite clauses
	index_dict["infinitive_prop"] = safe_divide(index_dict["to_clause"],index_dict["all_clauses"]) #infinitive clauses,total number of clauses
	index_dict["nonfinite_prop"] = safe_divide(index_dict["nonfinite_clause"],index_dict["all_clauses"]) #nonfinite clauses,total number of clauses
	

	return(index_dict)


def output_vertical(list_text,outname,ordered_output = "simple",prettyp = False): #list version of parsed text, list of attributes to output, name of output file
	if ordered_output == "full":
		ordered_output = ['idx','word','lemma','pos','tag','dep_rel','head','head idx','main_tag','spec_tag1','spec_tag2','spec_tag3','spec_tag4','spec_tag5','spec_tag6','semantic_tag1','semantic_tag2']
	elif ordered_output == "simple":
		ordered_output = ['idx','word','lemma','tag', 'dep_rel','head idx','main_tag','spec_tag1','spec_tag2','spec_tag3','spec_tag4','spec_tag5','spec_tag6','semantic_tag1','semantic_tag2']
	
	outf = open(outname,"w") #create output file
	for sent_id, sent in enumerate(list_text):
		if len(sent) < 1: #for some reason, there are some blank "sentences" in the output (at the end of each document)
			continue #skip these
		if sent_id != 0:
			outf.write("\n\n") #separate sentences by two line breaks
			if prettyp == True:
				print("\n\n")
		outf.write("#sentence " + str(sent_id))
		if prettyp == True:
			print("\n\n")
		for token in sent: #iterate through each token (dictionary) in sentence
			out_list = []
			for attr in ordered_output: #iterate through list of attributes to report
				if token[attr] == None:
					out_list.append("n/a") #add these to the list
				else:
					out_list.append(token[attr]) #add these to the list
			outf.write("\n" + "\t".join(out_list))
			if prettyp == True:
				print("\n" + "\t".join(out_list))
	outf.flush()
	outf.close()

def print_vertical(list_text,ordered_output = "simple"):
	if ordered_output == "full":
		ordered_output = ['idx','word','lemma','pos','tag','dep_rel','head','head idx','main_tag','spec_tag1','spec_tag2','spec_tag3','spec_tag4','spec_tag5','spec_tag6','semantic_tag1','semantic_tag2']
	elif ordered_output == "simple":
		ordered_output = ['idx','word','lemma','tag', 'dep_rel','head idx','main_tag','spec_tag1','spec_tag2','spec_tag3','spec_tag4','spec_tag5','spec_tag6','semantic_tag1','semantic_tag2']

	for sent_id, sent in enumerate(list_text):
		if len(sent) < 1: #for some reason, there are some blank "sentences" in the output (at the end of each document)
			continue #skip these
		if sent_id != 0:
			print("\n\n") #separate sentences by two line breaks
				
		print("#sentence " + str(sent_id))
		for token in sent: #iterate through each token (dictionary) in sentence
			out_list = []
			for attr in ordered_output: #iterate through list of attributes to report
				if token[attr] == None:
					out_list.append("n/a") #add these to the list
				else:
					out_list.append(token[attr]) #add these to the list
			print("\n" + "\t".join(out_list))

def output_xml(list_text,outname = False,xml_element = None):
	LGR_attr_list = ['main_tag','spec_tag1','spec_tag2','spec_tag3','spec_tag4','spec_tag5','spec_tag6','semantic_tag1','semantic_tag2'] #LGR specific tags
	if xml_element == None: #set xml_element if needed
		xml_element = ET.Element("tagged_text") #set root node in XML representation
	
	for sent_id, sent in enumerate(list_text):
		sent_level = ET.SubElement(xml_element,"sentence",attrib = {"sent_id":str(sent_id)}) 
		sent_text = ET.SubElement(sent_level,"sentence_text") #add xml tag for sentence text
		sentence_list = []
		for item in sent: #iterate through word dictionaries
			sentence_list.append(item["word"])
			wrd = ET.SubElement(sent_level,"word",attrib = {"idx":item["idx"]})
			raw_wrd = ET.SubElement(wrd,"raw") #create tag for raw words
			raw_wrd.text = item["word"] #add text
			
			lem = ET.SubElement(wrd,"lemma") #create tag for lemma
			lem.text = item["lemma"] #add lemma text
			btt = ET.SubElement(wrd,"biber_tags")
			for x in LGR_attr_list:
				if item[x] != None:
					btt.set(x,item[x])
			u_tag = ET.SubElement(wrd,"UPOS") #
			u_tag.text = item["pos"] #add Universal POS tag
			
			penn_pos = ET.SubElement(wrd,"POS")
			penn_pos.text = item["tag"] #add penn/spaCy POS tag
			
			head_rel = ET.SubElement(wrd,"DEP") #add head ID as attribute
			head_rel.text = item["dep_rel"] #add dependency relation head/governor
			head_rel.set("head",item["head"])
			head_rel.set("head_id", str(item["head idx"]))
		
		sent_text.text = " ".join(sentence_list) #add sentence to sentence_text tag
	
	if outname!= False:
		outf = open(outname,"w")
		outf.write(prettify(xml_element))
		outf.flush()
		outf.close()
	else:
		return(xml_element)

def sent_exampler(list_text,target):
	outl = []
	for sent_id, sent in enumerate(list_text):
		if len(sent) < 1: #for some reason, there are some blank "sentences" in the output (at the end of each document)
			continue #skip these
		s_text = []
		keep_s = False
		for token in sent: #iterate through each token (dictionary) in sentence
			#print(token)
			s_text.append(token["word"])
			for x in token:
				if target == token[x]:
					keep_s = True
					s_text.append("<--" + target + "<<<")
		if keep_s == True:
			outl.append(" ".join(s_text))
	return(outl)
							
def LGR_Full(filenames,outname,indices_dict=INDEX_LIST,cats_d = cats, outdirname = "", output = None): #output options should be list ["xml","vertical"]
	outf = open(outname,"w")#create output file
	outf.write("filename,"+",".join(indices_dict)) #write header
	if output != None: # if user specifies output type:
		if "xml" in output:
			xmldir = outdirname + "xml_output/"
			if os.path.exists(xmldir) == False: #if the folder doesn't exist, make it
				os.mkdir(xmldir)
		if "vertical" in output:
			vertdir = outdirname + "vertical_output/" 
			if os.path.exists(vertdir) == False: #if the folder doesn't exist, make it
				os.mkdir(vertdir)

	if type(filenames) == str: #allow filenames to be a list OR a target folder name/glob search
		if "*" not in filenames and filenames[-1] != "/": #if users forgot to include the "/"
			filenames = filenames + "/*.txt"
		if filenames[-1] == "/":
			filenames = filenames + "*.txt"
		filenames = glob.glob(filenames)
	
	for filename in filenames:
		simple_fname = filename.split("/")[-1] #grab the filename without all preceding folders
		print(simple_fname)
		text = open(filename).read()
		tag_output = LGR_Analysis(text,indices_dict,cats_d)
		output_list = [simple_fname]
		for x in indices_dict:
			if x in ["nwords","wrd_length"]:
				output_list.append(str(tag_output[x]))
			else:
				output_list.append(str((tag_output[x]/tag_output["nwords"])*10000)) #normed by 10,000 words
		outf.write("\n" + ",".join(output_list))

		#output annotated files
		if output != None:
			#output xml
			if "xml" in output:
				xmloutname = xmldir + "".join(simple_fname.split(".")[:-1]) + ".xml" #format filename
				output_xml(tag_output["tagged_text"],xmloutname,xml_element = None)
			#output vertical
			if "vertical" in output:
				vertoutname = vertdir + "".join(simple_fname.split(".")[:-1]) + ".tsv" #format filename
				output_vertical(tag_output["tagged_text"],vertoutname,ordered_output = "full") #write vertical output to file
	outf.flush()
	outf.close()

### process fix-tagged xml files
### Still need to deal with wrd_length and mattr. This will require tweaking the calcFromXml() function ###

#work on xmlreader
def calcFromXml(xml_filename,indices_dict=INDEX_LIST):
	simplefilename = xml_filename.split("/")[-1]
	index_dict = {}
	for x in indices_dict:
		index_dict[x] = 0 #start index counts
	tree = ET.parse(xml_filename)
	root = tree.getroot()
	for tags in root.iter("biber_tags"): #all biber tags are listed as tag attributes (e.g., <biber_tags main_tag="verb" spec_tag1="non_past_tense" semantic_tag1="mental_verb"/>)
		if len(tags.attrib) > 0:
			for x in tags.attrib:
				feature = tags.attrib[x]
				if feature in index_dict:
					index_dict[feature] += 1
				else:
					print("Warning! the tag <<<",feature,">>> is not a recognized tag. This may be due to typos in the tag-fixed files. Please double check the file <<<", simplefilename,">>>")
	for upos in root.iter("UPOS"): #all pos are listed as text within tags (e.g., <UPOS>NOUN</UPOS>)
		if upos.text not in ["PUNCT","SYM","SPACE","X"]:
			index_dict["nwords"] += 1
	
	return(index_dict)


def lgrXml(filenames,outname,indices_dict=INDEX_LIST):
	print(outname)
	outf = open(outname,"w")#create output file
	#need to deal with wrd_length and mattr. This will require tweaking the calcFromXml() function
	ignoreL = ["wrd_length","mattr","np","np_deps","relcl_dep","amod_dep","det_dep","prep_dep","poss_dep","cc_dep","all_clauses","finite_clause","finite_ind_clause","finite_dep_clause","finite_compl_clause","finite_relative_clause","nonfinite_clause","vp_deps","mean_nominal_deps","relcl_nominal","amod_nominal","det_nominal","prep_nominal","poss_nominal","cc_nominal","mean_verbal_deps","mlc","mltu","dc_c","ccomp_c","relcl_c"]
	INDEX_LIST = [x for x in indices_dict if x not in ignoreL]
	outf.write("filename,"+",".join(INDEX_LIST)) #write header
	for filename in filenames:
		tagDict = calcFromXml(filename,INDEX_LIST)
		simple_fname = filename.split("/")[-1] #grab the filename without all preceding folders
		print(simple_fname)
		output_list = [simple_fname]
		for x in INDEX_LIST:
			if x in ["nwords","wrd_length"]:
				output_list.append(str(tagDict[x]))
			else:
				output_list.append(str((tagDict[x]/tagDict["nwords"])*10000)) #normed by 10,000 words
		outf.write("\n" + ",".join(output_list))
	outf.flush()
	outf.close()

##############################################################################################################
### need to add a function for counting tags from fix-tagged files [xml first, then possibly vert as well] ###
##############################################################################################################

### USED IN Kyle et al 2021, 2022 ###
def LGR_XML(xml_files,outname,INDEX_LIST,cats): #for processing TMLE xml texts
	outf = open(outname,"w")#create output file
	ignore_list = "np np_deps relcl_dep amod_dep det_dep prep_dep poss_dep cc_dep all_clauses finite_clause finite_ind_clause finite_dep_clause finite_compl_clause finite_relative_clause nonfinite_clause vp_deps".split(" ")
	
	refined_INDEX_LIST = [x for x in INDEX_LIST if x not in ignore_list] #ignores raw counts for complexity items
	outf.write("filename,learning_environment,mode,discipline,subdiscipline,text_type,"+",".join(refined_INDEX_LIST)) #write header
	
	tt_list = open("lists_LGR/text_type_map_2020-5-24.txt").read().split("\n")
	tt_dict = {}
	for x in tt_list:
		line = x.split("\t")
		tt_dict["\t".join([line[0],line[1],line[2]])] = line[3]
	
	def discipline_fixer(discipline):
		typo_dict = {"natural_sciences":"natural_science","natual_science":"natural_science","anthropology":"humanities","social_sciences":"social_science","marketing":"business","astronomy":"natural_science","english":"humanities","chemistry":"natural_science","pnatural_science":"natural_science","n/a":"service_encounters"}
		if len(discipline.split(" ")) == 1:
			dp = discipline.lower()
		elif discipline.split(" ")[0].lower() == "social":
			dp = "social_science"
		elif discipline.split(" ")[0].lower() == "natural":
			dp = "natural_science"
		if dp in typo_dict:
			dp = typo_dict[dp]
		return(dp)
		
	def cleaner(thingy):
		cleaned = thingy.replace(",","_")
		cleaned = cleaned.replace(" ","_")
		return(cleaned)
		
	for filename in xml_files:
		simple_fname = filename.split("/")[-1] #grab the filename without all preceding folders
		print(simple_fname)
		#if simple_fname in ["socpooh___n149.xml","10206_9-30-2019_19-11-38.xml","10200_9-25-2019_13-51-02.xml","10062_10-24-2018_10-42-32.xml","HarvardX_QMB1_06-28-2020_00-00-38.xml","HarvardX_QMB1_06-28-2020_00-00-39.xml","DartmouthX_RFundX_12-18-2020_00-00-26.xml","HarvardX_QMB1_06-28-2020_00-00-62.xml","HarvardX_QMB1_06-28-2020_00-00-63.xml","10052_10-29-2018_9-15-50.xml","HarvardX_QMB1_06-28-2020_00-00-61.xml","10150_2-26-2019_7-15-00.xml","10109_3-2-2019_8-42-04.xml","MichiganX_ArtsAdminx_02-19-2021_00-00-19_02.xml"]:#skipe these for now - need to update other code and/ or check files
			#continue
		tree = ET.parse(filename)
		root = tree.getroot()
		if "learning_environment" not in root[0].attrib:
			le = "tmle"
		else: le = "traditional"
		
		if le != "traditional":
			if root[0].attrib["provided_by"] == "student":
				continue
		if "subdiscipline" not in root[0].attrib:
			if "subject" in root[0].attrib:
				sdp = cleaner(root[0].attrib["subject"])
			else: sdp = "n/a" #this is because some files don't have a subject - this is a problem
		else: sdp = cleaner(root[0].attrib["subdiscipline"])
		pre_tt = "\t".join([le,root[0].attrib["mode"],cleaner(root[0].attrib["file_type"])])
		output_list = [simple_fname,le,root[0].attrib["mode"],discipline_fixer(root[0].attrib["discipline"]),sdp,tt_dict[pre_tt]] #start list for indices
		
		if root[1].attrib["text_type"] in ["plain_text","plaintext"]:
			text = root[1].text
		else:
			if len(root) < 3: #this is due to a problem. Some texts have some issue
				print(simple_fname + "\ttext tag problem")
				continue
			else:
				text = root[2].text
		
		output = LGR_Analysis(text,INDEX_LIST,cats)
		no_norming = "nwords wrd_length mattr mean_nominal_deps relcl_nominal amod_nominal det_nominal prep_nominal poss_nominal cc_nominal mean_verbal_deps mlc mltu dc_c ccomp_c relcl_c infinitive_prop nonfinite_prop".split(" ")
		for x in refined_INDEX_LIST:
			if x in no_norming:
				output_list.append(str(output[x]))
			else:
				output_list.append(str((output[x]/output["nwords"])*10000)) #normed by 10,000 words
		outf.write("\n" + ",".join(output_list))
	
	outf.flush()
	outf.close()

### IN Progress ###
def Simple_XML_Reader(xml_files,INDEX_LIST,cats,target):
	ex_sents = []
	for filename in xml_files:
		simple_fname = filename.split("/")[-1] #grab the filename without all preceding folders
		print(simple_fname)
		tree = ET.parse(filename)
		root = tree.getroot()
		
		if root[1].attrib["text_type"] in ["plain_text","plaintext"]:
			text = root[1].text
		else:
			if len(root) < 3: #this is due to a problem. Some texts have some issue
				continue
			else:
				text = root[2].text
		ex_sents += sent_exampler(LGR_Analysis(text,INDEX_LIST,cats,output = True)["tagged_text"],target)
	
	return(ex_sents)
		
##########################################
	
### Other functions #####
def LGR_tt_find(xml_files): #for processing TMLE xml texts
	tt_list = open("lists_LGR/text_type_map.txt").read().split("\n")
	tt_dict = {}
	for x in tt_list:
		line = x.split("\t")
		tt_dict[line[0]] = line[1]
	
	def cleaner(thingy):
		cleaned = thingy.replace(",","_")
		cleaned = cleaned.replace(" ","_")
		return(cleaned)
		
	for filename in xml_files:
		simple_fname = filename.split("/")[-1] #grab the filename without all preceding folders
		#print(simple_fname)
		#if simple_fname in ["socpooh___n149.xml","10206_9-30-2019_19-11-38.xml","10200_9-25-2019_13-51-02.xml","10062_10-24-2018_10-42-32.xml","HarvardX_QMB1_06-28-2020_00-00-38.xml","HarvardX_QMB1_06-28-2020_00-00-39.xml","DartmouthX_RFundX_12-18-2020_00-00-26.xml","HarvardX_QMB1_06-28-2020_00-00-62.xml","HarvardX_QMB1_06-28-2020_00-00-63.xml","10052_10-29-2018_9-15-50.xml","HarvardX_QMB1_06-28-2020_00-00-61.xml","10150_2-26-2019_7-15-00.xml","10109_3-2-2019_8-42-04.xml","MichiganX_ArtsAdminx_02-19-2021_00-00-19_02.xml"]:#skipe these for now - need to update other code and/ or check files
			#continue
		tree = ET.parse(filename)
		root = tree.getroot()
		
		if cleaner(root[0].attrib["file_type"]) in tt_dict:
			continue
		else:
			print(cleaner(root[0].attrib["file_type"])) #start list for indices

def LGR_discipline_check(xml_files):
	def discipline_fixer(discipline):
		if len(discipline.split(" ")) == 1:
			dp = discipline.lower()
		elif discipline.split(" ")[0].lower() == "social":
			dp = "social_science"
		elif discipline.split(" ")[0].lower() == "natural":
			dp = "natural_science"
		return(dp)

	discipline_dict = {}
	for filename in xml_files:
		simple_fname = filename.split("/")[-1] #grab the filename without all preceding folders
		tree = ET.parse(filename)
		root = tree.getroot()
		discipline = discipline_fixer(root[0].attrib["discipline"])
		if discipline in discipline_dict:
			discipline_dict[discipline] += 1
		else:
			discipline_dict[discipline] = 1
	return(discipline_dict)

def clean_text(in_text):
	
	if "[" in in_text and "]" in in_text:
		in_text = re.sub("\[.*?\]","",in_text)
	if "1:" in in_text:
		in_text = re.sub("\n[0-9]:","",in_text)
	if " " in in_text:
		if "\n" not in in_text:
			in_text = " ".join(in_text.split())
		else:
			line_text = []
			for x in in_text.split("\n"):
				line_text.append(" ".join(x.split()))
			in_text = "\n".join(line_text)
	return(in_text)
	
#####				

# Data Analysis Starts Here
###########################################

# try2 = LGR_Full(glob.glob("practice/*.txt"),"practice_results.csv",INDEX_LIST,cats)

# #Try with XML:
# try3 = LGR_XML(glob.glob("TMLE_sample/*.xml") + glob.glob("T2KSWAL/*/*.xml"),"xml_results.csv",INDEX_LIST,cats)

	
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),25),INDEX_LIST,cats,"wh_relative_subj_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"wh_relative_obj_clause"),5)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"wh_relative_prep_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"past_participial_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"perfect_aspect"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"agentless_passive"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"by_passive"),10)
# 
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"that_complement_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"that_verb_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"that_adjective_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"that_noun_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"to_clause"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"to_clause_noun"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"to_clause_verb"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"to_clause_adjective"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"jj_attributive"),10)
# 
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"pp_demonstrative"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"pv_do"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"complementizer_that0"),20)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"split_aux"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"cc_phrase"),10)
# sample(Simple_XML_Reader(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml"),50),INDEX_LIST,cats,"cc_clause"),10)

# #run data 2020-5-26 - All TMLE files and all T2KSWAL Files
# test1 = LGR_XML(sample(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml") + glob.glob("T2KSWAL_2020-5-22/*/*.xml"),25),"xml_results_LGR58_2020-5-26_test.csv",INDEX_LIST,cats)
# test2 = LGR_XML(glob.glob("TMLE_2020-5-16/TMLE_2020-5-16/encountered/*.xml") + glob.glob("T2KSWAL_2020-5-22/*/*.xml"),"xml_results_LGR58_2020-5-26.csv",INDEX_LIST,cats)


# ### Various tests ###

# try4 = LGR_Analysis("I think that we should fire him. He has, after all, caused a lot of problems.", INDEX_LIST,cats)

# try6 = LGR_Analysis("I think that we should fire him. He has, after all, caused a lot of problems that", INDEX_LIST,cats)
# try6 = LGR_Analysis("The manner in which he was told was cruel.", INDEX_LIST,cats)

# ex_tester("the man who killed harry is here")
# ex_tester("the man that killed harry is here")
# ex_tester("That is a red hat")
# ex_tester("He is a teacher")
# ex_tester("That is his car")
# ex_tester("That is his car or his wife")

# ex_tester("I mow the lawn and she pulls the weeds")
# ex_tester("I mow the lawn and pull the weeds")
# ex_tester("He runs and jumps")
# ex_tester("I like to run")
# ex_tester("Running is what I like")
# ex_tester("matter  seeing those problems again, and that's just one, one set of problems.")
# ex_tester("I have been trying this for years")
# ex_tester("He has been killed")
# ex_tester("The manner in which he was killed was deplorable")
# ex_tester("the road through which we must pass has killed many")
# ex_tester("we will go wherever you want to go")
# ex_tester("we will kill whomever you want us to kill")
# ex_tester("We will go wherever we want")
# ex_tester("I have been trying this forever")
# ex_tester("Sometimes an entire plan had to be redone .")
# ex_tester("Where we are going to we don't need anything.")
# ex_tester("They are people I have worked with.")
# ex_tester("Covers damage to or loss of your home and possessions")
# ex_tester("Fusion reactions , compared to fission reactions")
# ex_tester("We stopped at a rest stop and got out of the car")
# ex_tester("All other course materials ( lecture slides , problem sets , handouts , etc ) will be posted on Laulima - please check Laulima regularly")
# ex_tester('For other pairs , the correlation is in between -1 and 1 .')
# ex_tester("I put all the operations that operate on x")
# ex_tester("Like the other obligations of justice already spoken of, this one is not regarded as absolute")
# ex_tester('In general this seems to be correct assessment for the majority of Korean peasants .')
# ex_tester('Ms. Jung is particularly concerned about the amount of student movement in and out of the classroom required throughout the day .')
# ex_tester("Established over a century ago by France ’s cooperative sector to finance cooperatives")
# ex_tester("Purpose Young adult literature ( YAL ) has the power to bring to light common experiences , emotions , and dilemmas while also promoting respect for differences")
# ex_tester('So they said look , we really have to reorder things as we sell them .')
# ex_tester('So they said look , we really have to reorder things as we sell them .')


#clean_text(sample).strip()
