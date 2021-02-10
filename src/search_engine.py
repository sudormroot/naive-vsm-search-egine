#!/usr/bin/env python3

import numpy as np
import string
import sqlite3
import os


doc1="""
O wild West Wind, thou breath of Autumn's being,
Thou, from whose unseen presence the leaves dead
Are driven, like ghosts from an enchanter fleeing,

Yellow, and black, and pale, and hectic red,
Pestilence-stricken multitudes: O thou,
Who chariotest to their dark wintry bed

The winged seeds, where they lie cold and low,
Each like a corpse within its grave, until
Thine azure sister of the Spring shall blow

Her clarion o'er the dreaming earth, and fill
(Driving sweet buds like flocks to feed in air)
With living hues and odours plain and hill:

Wild Spirit, which art moving everywhere;
Destroyer and preserver; hear, oh hear!
"""

doc2="""
Thou on whose stream, mid the steep sky's commotion,
Loose clouds like earth's decaying leaves are shed,
Shook from the tangled boughs of Heaven and Ocean,

Angels of rain and lightning: there are spread
On the blue surface of thine aëry surge,
Like the bright hair uplifted from the head

Of some fierce Maenad, even from the dim verge
Of the horizon to the zenith's height,
The locks of the approaching storm. Thou dirge

Of the dying year, to which this closing night
Will be the dome of a vast sepulchre,
Vaulted with all thy congregated might

Of vapours, from whose solid atmosphere
Black rain, and fire, and hail will burst: oh hear!
"""

doc3="""
Thou who didst waken from his summer dreams
The blue Mediterranean, where he lay,
Lull'd by the coil of his crystalline streams,

Beside a pumice isle in Baiae's bay,
And saw in sleep old palaces and towers
Quivering within the wave's intenser day,

All overgrown with azure moss and flowers
So sweet, the sense faints picturing them! Thou
For whose path the Atlantic's level powers

Cleave themselves into chasms, while far below
The sea-blooms and the oozy woods which wear
The sapless foliage of the ocean, know

Thy voice, and suddenly grow gray with fear,
And tremble and despoil themselves: oh hear!
"""

doc4="""
If I were a dead leaf thou mightest bear;
If I were a swift cloud to fly with thee;
A wave to pant beneath thy power, and share

The impulse of thy strength, only less free
Than thou, O uncontrollable! If even
I were as in my boyhood, and could be

The comrade of thy wanderings over Heaven,
As then, when to outstrip thy skiey speed
Scarce seem'd a vision; I would ne'er have striven

As thus with thee in prayer in my sore need.
Oh, lift me as a wave, a leaf, a cloud!
I fall upon the thorns of life! I bleed!

A heavy weight of hours has chain'd and bow'd
One too like thee: tameless, and swift, and proud.
"""

doc5="""
Make me thy lyre, even as the forest is:
What if my leaves are falling like its own!
The tumult of thy mighty harmonies

Will take from both a deep, autumnal tone,
Sweet though in sadness. Be thou, Spirit fierce,
My spirit! Be thou me, impetuous one!

Drive my dead thoughts over the universe
Like wither'd leaves to quicken a new birth!
And, by the incantation of this verse,

Scatter, as from an unextinguish'd hearth
Ashes and sparks, my words among mankind!
Be through my lips to unawaken'd earth

The trumpet of a prophecy! O Wind,
If Winter comes, can Spring be far behind?
"""

# All documents
docs={"doc1":doc1, "doc2":doc2, "doc3":doc3, "doc4":doc4, "doc5":doc5}


for d in docs:
    print("document: ", d)
    print(docs[d])



#if not os.path.exists("data"):
#    os.makedirs("data")


# create database for storing metadata
#conn = sqlite3.connect('data/naive-vsm-search-engine.db')


# Tokenizing and pre-processing documents
tokened_docs={}

# unique vocabulary
vocab = []

stop_words=["of", "in", "on", "is"]

for k,v in docs.items():

    print("Pre-process document: ", k)
    tokens = v.split()

    # to lower
    tokens = [w.lower() for w in tokens]

    # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    tokens = [w for w in tokens if w not in string.punctuation]

    # remove stop words
    tokens = [w for w in tokens if w not in stop_words]

    #print(tokens)
    
    tokened_docs[k] = tokens
    
    vocab = vocab + tokens

# get unique words
vocab = set(vocab)

print("Unique words:")
print(vocab)

print("")
print("unique words length: ", len(vocab))
print("")



# We create vectors for documents here
vectors = {}

for doc, tokens in tokened_docs.items():
    print("Calculate vector for document: ", doc)

    v = []

    for w in vocab:
        v.append(tokens.count(w))

    print("v=", v)

    vectors[doc] = np.array(v)


while True:
    print("\n")
    print("\n")
    q = input("Please input a query (q=quit):")
    print("q=",q)

    if q == "q":
        break

    # Tokenize query
    q = q.split()

    q = [w for w in q if w in vocab]

    print("Normalized query: ", q)

    # Calculate query vector
    vq = []
    for w in vocab:
        vq.append(q.count(w))

    print("vq=", vq)

    # prevent the divisor is zero.
    if np.sum(vq) == 0:
        vq[0] = 0.0001

    vq = np.array(vq)

    # formulae of cosine similarity
    sim = lambda d, q: np.sum(d * q) / np.sqrt(np.sum(d**2) * np.sum(q**2))

    scores={}
    scores_sum = 0.

    # calculate similarities for all documents
    for doc, vd in vectors.items():
        s = sim(vd, vq)
        print(f"sim({doc}, query)={s:.5f}")
        scores[doc] = s
        scores_sum += s

    print("")

    # normalize scores
    for doc, score in scores.items():
        scores[doc] = scores[doc]/scores_sum
        print(f"Normalized score of {doc}={scores[doc]:.5f}")


    print("")
