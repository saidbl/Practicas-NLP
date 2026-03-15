import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def remove_contents(text):
    if "CONTENTS" in text:
        text = text.split("CONTENTS",1)[1]
    return text

def cut_story_end(text):
    end_marker = "THE END."
    pos = text.find(end_marker)
    if pos != -1:
        text = text[:pos + len(end_marker)]
    return text

def clean_text(text):
    text = remove_contents(text)
    text = re.sub(r"\[Illustration.*?\]", " ", text)
    text = re.sub(r"_Heading_", " ", text)
    text = re.sub(r"_Full-page design_", " ", text)
    text = re.sub(r"_Half-page design_", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def find_story_positions(text, titles):
    positions = []
    for title in titles:
        pattern = re.compile(re.escape(title) + r"\.", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            positions.append((title, match.start()))
    positions = sorted(positions, key=lambda x: x[1])
    return positions

def split_stories(text, titles):
    text = clean_text(text)
    positions = find_story_positions(text, titles)
    stories = {}
    for i in range(len(positions)):
        title, start = positions[i]
        if i < len(positions)-1:
            end = positions[i+1][1]
        else:
            end = len(text)
        story = text[start:end]
        story = cut_story_end(story)
        stories[title] = story.strip()
    return stories

def remove_short_stories(stories, min_tokens=200):
    filtered = {}
    for title, text in stories.items():
        if len(text.split()) > min_tokens:
            filtered[title] = text
    return filtered

def inspect_stories(stories):
    for i,(title,text) in enumerate(stories.items(),1):
        tokens = len(text.split())
        print("\nStory",i)
        print("TITLE:",title)
        print("TOKENS:",tokens)
        print("CHARACTERS:",len(text))
        print("START:",text[:150])
        print("END:",text[-150:])
        print("-"*60)
with open("fairyBook.txt","r",encoding="utf8") as f:
    book1 = f.read()
with open("salvePeasant.txt","r",encoding="utf8") as f:
    book2 = f.read()
titles_book1 = [
"THE SLEEPING BEAUTY IN THE WOOD",
"HOP-O'-MY-THUMB",
"CINDERELLA; OR, THE LITTLE GLASS SLIPPER",
"ADVENTURES OF JOHN DIETRICH",
"BEAUTY AND THE BEAST",
"LITTLE ONE EYE, LITTLE TWO EYES, AND LITTLE THREE EYES",
"JACK THE GIANT KILLER",
"TOM THUMB",
"RUMPELSTILZCHEN",
"FORTUNATUS",
"THE BREMEN TOWN MUSICIANS",
"RIQUET WITH THE TUFT",
"HOUSE ISLAND",
"SNOW-WHITE AND ROSE RED",
"JACK AND THE BEAN-STALK",
"GRACIOSA AND PERCINET",
"THE IRON STOVE",
"THE INVISIBLE PRINCE",
"THE WOODCUTTER'S DAUGHTER",
"BROTHER AND SISTER",
"LITTLE RED-RIDING-HOOD",
"PUSS IN BOOTS",
"THE WOLF AND THE SEVEN YOUNG GOSLINGS",
"THE FAIR ONE WITH GOLDEN LOCKS",
"THE BUTTERFLY",
"THE FROG-PRINCE",
"THE WHITE CAT",
"PRINCE CHERRY",
"LITTLE SNOWDROP",
"THE BLUE BIRD",
"THE YELLOW DWARF",
"THE SIX SWANS",
"THE PRINCE WITH THE NOSE",
"THE HIND OF THE FOREST",
"THE JUNIPER TREE",
"CLEVER ALICE"
]
titles_book2 = [
"THE ABODE OF THE GODS",
"THE SUN; OR, THE THREE GOLDEN HAIRS OF THE OLD MAN VSEVEDE",
"KOVLAD",
"THE MAID WITH HAIR OF GOLD",
"THE JOURNEY TO THE SUN AND THE MOON",
"THE DWARF WITH THE LONG BEARD",
"THE FLYING CARPET, THE INVISIBLE CAP, THE GOLD-GIVING RING, AND THE SMITING CLUB",
"THE BROAD MAN, THE TALL MAN, AND THE MAN WITH EYES OF FLAME",
"THE HISTORY OF PRINCE SLUGOBYL; OR, THE INVISIBLE KNIGHT",
"THE SPIRIT OF THE STEPPES",
"THE PRINCE WITH THE GOLDEN HAND",
"IMPERISHABLE",
"OHNIVAK",
"TEARS OF PEARLS",
"THE SLUGGARD",
"KINKACH MARTINKO",
"THE STORY OF THE PLENTIFUL TABLECLOTH, THE AVENGING WAND, THE SASH THAT BECOMES A LAKE, AND THE TERRIBLE HELMET"
]
stories1 = split_stories(book1, titles_book1)
stories2 = split_stories(book2, titles_book2)
stories1 = remove_short_stories(stories1)
stories2 = remove_short_stories(stories2)
print("Stories in Book 1:", len(stories1))
print("Stories in Book 2:", len(stories2))
print("\n========== BOOK 1 ==========")
inspect_stories(stories1)
print("\n========== BOOK 2 ==========")
inspect_stories(stories2)
docs1 = [preprocess(x) for x in stories1.values()]
docs2 = [preprocess(x) for x in stories2.values()]
docs = docs1 + docs2
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
A = X[:len(docs1)]
B = X[len(docs1):]
sim = cosine_similarity(A,B)
results = []
titles1 = list(stories1.keys())
titles2 = list(stories2.keys())
for i in range(len(titles1)):
    for j in range(len(titles2)):
        results.append({
            "Story from Book 1": titles1[i],
            "Story from Book 2": titles2[j],
            "Similarity Score": sim[i,j]
        })
df = pd.DataFrame(results)
top5 = df.sort_values("Similarity Score", ascending=False).head(5)
print("\nTop 5 Most Similar Stories\n")
print(top5)

