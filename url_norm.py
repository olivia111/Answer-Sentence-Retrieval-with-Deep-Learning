import re
from url_normalize import url_normalize
import pandas as pd
import csv

import string
# from util import remove_part_in_string

tag_b = re.compile(r"<b>|</b>")
tag_html = re.compile(r"<[/]?.*?>")
tag_html_keep = re.compile(r"(<[/]?.*?>)")
tag_div = re.compile(r"<div[^<>]+>")
tag_omit = re.compile(r"\.\.\.")
tag_spaces = re.compile(r"[ ]+")
tag_twomore_spaces = re.compile(r" [ ]+")
tag_list_start_end = re.compile(r"(<ol>.*</ol>)|(<ul>.*</ul>)")
tag_bullet_no_end = re.compile(r"<li>.*")
tag_bullet = re.compile(r"<li>.*?</li>")
tag_punctuation = re.compile(r"[.?\-,()\[\]!]")
tag_alphabet_number = re.compile(r"[^"+ string.ascii_lowercase + string.digits + "]")


string_period_pattern = re.compile(r"[%s]\.[%s]" %(string.ascii_lowercase, string.ascii_uppercase))




def remove_part_in_string(pattern, s):

    m = pattern.search(s)
    # print(m)
    if m:
        # print(m.span())
        # print(m.string)
        s = s[0: m.start()] + s[m.end():]
        # s = s.replace(m.string, "")
    return s

def remove_all_matched_parts_in_string(pattern, s):

    return re.sub(pattern, "", s)

def remove_all_matched(pattern, s):
    return remove_all_matched_parts_in_string(pattern, s)

def merge_spaces(s):
    return tags.tag_twomore_spaces.sub(" ", s)

def remove_all_matched_except(pattern, s, l):
    # print("before ", s)
    def fun(m):
        if m == None:
            return ""

        if m.group() not in l:
            return ""
        else:
            return m.group()
    s = re.sub(pattern, fun, s)
    # print("after ", s)
    return s





def oneside_similarity_Jaccard(passage, gold_passage):

    word_set1 = set(passage.split(" "))
    word_set2 = set(gold_passage.split(" "))
    num_commonwords = len(word_set1.intersection(word_set2))
    # num_distinctwords = len(word_set1.union(word_set2))

    return float(num_commonwords) / float(len(word_set1))



def normalize_text(text):

    #define patterns
    #add more pattern here
    #pattern 1
    def pattern_1(s):
        def period_fun(m):

            print(m.group())
            s_t = m.group().replace('.', ' . ')
            print(s_t)
            return s_t

        s = string_period_pattern.sub(period_fun, s)
        return s

    text = text.replace("“", "\"").replace("”", "\"").replace("``", "\"").replace("''", "\"").replace("’", "'")
    text = pattern_1(text)

    text = text.lower()

    return text



def clean_url(url):
    url = url_normalize(url)
    url = url.replace("&amp;", "").replace("&","")
    url = url.strip()
    p = re.compile(r"http[s]?://(www.)?")
    url = p.sub("", url)
    #remove the last /
    url = url.strip()
    if url[-1] == '\/':
        url = url[0:len(url) - 1]

    return url

def normalize_url(url):
    return url_normalize(url)


if __name__ == "__main__":
    pos = pd.read_csv(r"C:\Users\silin\Documents\Tric\data\Models_jan04_part1.Neg.M30.txt", header=0, quoting=csv.QUOTE_NONE,
                      delimiter="\t", error_bad_lines=False)
    pos.columns = ['query_token', 'url', 'id','chunk']
    pos['url_clean'] = pos.url.apply(clean_url)
    pos.to_csv(r"C:\Users\silin\Documents\Tric\data\Models_jan04_part1.Neg.M30_clean.txt", sep='\t', encoding="utf8")



