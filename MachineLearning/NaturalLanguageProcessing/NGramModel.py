import random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests

data =  [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
          ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
          ("data science", 60, 70), ("analytics", 90, 3), ("team player", 85, 85),
          ("dynamic", 2, 90), ("synergies", 70, 0), ("actionable insights", 40, 30), ("think out of the box", 45, 10),
          ("self-starter", 30, 50), ("customer focus", 65, 15), ("thought leadership", 35, 35)]

def fix_unicode(text):
    return text.replace(u"\u2019", "'")

def get_document():

    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')

    content = soup.find("div", "article-body")  # find article-body div
    regex = r"[\w']+|[\.]" # matches a word or a period

    document = []

    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document

"""
Bigram Model
"""

def generateTransitions(document):
    bigrams = zip(document, document[1:])
    transitions = defaultdict(list)
    for prev, current in bigrams:
        transitions[prev].append(current)

    return transitions

def generate_using_bigrams(transitions):
    current = "."   # this means the next word will start a sentence
    result = []
    while True:
        next_word_candidates = transitions[current]    # bigrams (current, _)
        current = random.choice(next_word_candidates)  # choose one at random
        result.append(current)                         # append it to results
        if current == ".": return " ".join(result)     # if "." we're done

"""
Trigram Model
"""

def generateTrigramTransitions(document):
    trigrams = zip(document, document[1:], document[2:])
    trigram_transitions = defaultdict(list)
    starts = []

    for prev, current, next in trigrams:
        if prev == ".":              # if the previous "word" was a period
            starts.append(current)   # then this is a start word

        trigram_transitions[(prev, current)].append(next)

    return starts, trigram_transitions

def generate_using_trigrams(starts, trigram_transitions):
    current = random.choice(starts)   # choose a random starting word
    prev = "."                        # and precede it with a '.'
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next = random.choice(next_word_candidates)

        prev, current = current, next
        result.append(current)

        if current == ".":
            return " ".join(result)

"""
Grammer Model
"""

def generate_sentence(grammar):
    return expand(grammar, ["_S"])

def is_terminal(token):
    return token[0] != "_"

def expand(grammar, tokens):
    for i, token in enumerate(tokens):

        if is_terminal(token): continue

        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        return expand(grammar, tokens)

    return tokens



if __name__ == "__main__":
    print("Loading data")
    doc = get_document()
    print("Data loaded")

    print("Generating Bigrams")
    transitions = generateTransitions(doc)
    sentence = generate_using_bigrams(transitions)
    print(sentence)
    print()

    print("Generating Trigrams")
    starts, transitions = generateTrigramTransitions(doc)
    sentence = generate_using_trigrams(starts, transitions)
    print(sentence)
    print()

    print("Generating sentence using grammer")
    grammar = {    "_S"  : ["_NP _VP"],
                   "_NP" : ["_N","_A _NP _P _A _N"],
                   "_VP" : ["_V", "_V _NP"],
                   "_N"  : ["data science", "Python", "regression"],
                   "_A"  : ["big", "linear", "logistic"],
                   "_P"  : ["about", "near"],
                   "_V"  : ["learns", "trains", "tests", "is"] }

    # _ represents rule, without represents terminals

    sentence = generate_sentence(grammar)
    print(sentence)