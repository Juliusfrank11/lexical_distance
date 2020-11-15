import unidecode as udc
import pandas as pd
from difflib import SequenceMatcher
import Levenshtein
#imports for wikitionary scaping, not used now
#import mwparserfromhell
#import requests
#import re
from math import log
import numpy as np
from googletrans import Translator
from sklearn.cluster import AffinityPropagation as ap
import time
import wordcloseness_constants as wc_con

#Old wikitionary webscraping version
#def wiki_translate(lang,title):
#    if lang == 'English':
#        return title.replace('_',' ').lower()
#    else:
#        try:
#            if lang == 'Serbo-Croatian':
#                lang = 'Roman' #easy way to deal with the two different alphabets, since Roman isn't a language
#            params = {
#            "action": "query",
#            "prop": "revisions",
#            "rvprop": "content",
#            "rvslots": "main",
#            "rvlimit": 1,
#            "titles": title,
#            "format": "json",
#            "formatversion": "2",
#            }
#            headers = {"User-Agent": "My-Bot-Name/1.0"}
#            req = requests.get("https://en.wiktionary.org/w/api.php", headers=headers, params=params)
#            res = req.json()
#            revision = res["query"]["pages"][0]["revisions"][0]
#            text = revision["slots"]["main"]["content"]
#            wikicode = str(mwparserfromhell.parse(text))
#            translations = wikicode[wikicode.index('====Translations===='):wikicode.index('{{trans-bottom}}')]
#            translation = translations[translations.index(lang + ':'):translations.index('*',translations.index(lang + ':'))]
#            if 'tr=' in translation:
#                pattern = re.compile(r'[\,\}\|]')
#                t = udc.unidecode(translation[translation.index('tr=')+3:pattern.search(translation,translation.index('tr=')+3).start()]).strip()
#            else:
#                t = udc.unidecode(translation[:translation.index('}}')].split('|')[2])
#            return t.lower()
#        except ValueError:
#            print('No translation for {0} in {1}'.format(title,lang))
#            return None

def google_translate(lang,word):
    #convert to abbrivations
    if lang not in wc_con.lang_dict.keys():
        lang = wc_con.inv_lang_dict[lang]
    if lang == 'en':
        return word.lower()
    if lang == 'fa': #persian translates extremely badly so don't waste time, but I need to keep the persian columns to keep consistency
    # If I didn't do this it would be spending an hour overall just translating persian
        return 'persian translates extremely badly, see README'
    t = Translator()
    r = t.translate(word,dest=lang,src='en')
    # pronunciation is weird, it's supposed to giver transliteration of non-latin scripts
    # But for languages written in the Latin script, it sometimes gives None and sometimes gives the English word
    # this conditional catches both
    if r.pronunciation != word and r.pronunciation is not None:
        a = udc.unidecode(r.pronunciation)
    else:
        a = udc.unidecode(r.text)
    return a.lower()


def compare_sim(w,y):
    #old leftover from wikitionary code
    if w is None or y is None:
        return None
    ratings = []
    ratings.append(SequenceMatcher(None,w,y).ratio())
    ratings.append(Levenshtein.ratio(w,y))
    rating = sum(ratings)/len(ratings)
    if rating == 0 or rating > 4:
        #I've never seens a rating of 4 or higher naturally, such a rathing would have a ratio of 0.0625 or lower
        #Thus I opt to make 4 the cut off point for complete dissimilarity  
        return 4
    elif rating == 1:
        #will return -0.0 if this is not stated
        return 0
    else:
        # want to penialize farther apart words
        return -log(rating,2)

def gen_comp_matrix(word,langs,include_word=True,p=False,classify=True,include_ruler=True,sleep=1):
    # p is just an option to provide status updates via console
    # ruler, classify, and word include the colonial ruler, language family, and word of the language respectively
    # sleep tells the program to wait sleep seconds between translations, so google doesn't think we're DDoSing them
    word = word.lower()
    if p:
        print('Starting translation of',word)
    num = len(langs)
    A = np.zeros((num,num))
    done = []
    #do translating here to save time
    translated = {}
    for lang in langs:
        if lang not in ('fa','en'): #don't need to sleep for english and persian
            time.sleep(sleep) # please don't ban me
        r = google_translate(lang,word)
        translated[lang] = r
        if p:
            print('Done translating',word,'to',wc_con.lang_dict[lang].capitalize(),':',r)
            #check for too many values that are the same as the english word, indicative of falled translations
        if len([i for i in translated.values() if i == word]) > 15:
            print('skipping',word,'too many similar words')
            return None
    #check for too many simular values, indicative of falled translations
    if p:
        print('Comparing',word,'in different languages',end='')
    for n in range(num):
        for m in range(num):
            if n == m or m in done:
                continue
            else:
                # This cuts computation time by about half by filling both entries at once
                rating = compare_sim(translated[langs[n]],translated[langs[m]])
                A[n][m] = A[m][n] = rating
                if n not in done:
                    done.append(n)
                if p:
                    print('.',end='')
    if p:
        print()
        print('Finished comparing',word,end='!\n')
    labels = [wc_con.lang_dict[l] for l in langs]
    # want to add language family information?
    if not classify and not include_word:
        return pd.DataFrame(data=A,index=labels,columns=labels)
    else:
        df = pd.DataFrame(data=A,index=labels,columns=labels)
        #want to include the word?
        if include_word:
            df['word'] = [translated[l] for l in langs]
        # want to add language family information?  
        if classify:
            df['classification'] = [wc_con.lang_fam_dict[l] for l in langs]
        if include_ruler:
            df['ruler'] = [wc_con.lang_ruler_dict_main[l] for l in langs]
        return df
    
def get_clusters(df,damping=0.5,include_word=False):
    aff = ap(damping=damping)
    aff.fit(df)
    clus = aff.predict(df)
    cluster_dict = {}
    for i in range(len(set(clus))):
        cluster_dict[i] = []
    for cluster, n in zip(clus,range(len(clus))):
        cluster_dict[cluster].append((df.index[n]))
    return cluster_dict

def get_clusters_with_word(df,damping=0.5):
    aff = ap(damping=damping)
    aff.fit(df.drop('word',axis=1))
    clus = aff.predict(df.drop('word',axis=1))
    cluster_dict = {}
    for i in range(len(set(clus))):
        cluster_dict[i] = []
    for cluster, lang, n in zip(clus,df,range(len(clus))):
        cluster_dict[cluster].append((df.index[n],df.loc[lang,'word']))
    return cluster_dict

def print_cluster(cluster_dict):
    for cluster in cluster_dict.keys():
        print('Cluster #',cluster)
        for lang in cluster_dict[cluster]:
            print(lang)
        print()
            