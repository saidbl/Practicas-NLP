import re
import nltk
from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english')) | set(stopwords.words('spanish'))

with open("text.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+\b', text)

filtered_words = [word for word in words if word not in stop_words]

word_count = Counter(filtered_words)

top_words = word_count.most_common(100)

for word, count in top_words:
    print(word, count)

from nltk.probability import FreqDist

frequency_words = FreqDist(filtered_words)

print("\nLas 100 palabras más frecuentes (excluyendo stopwords):")
for word, frequency in frequency_words.most_common(100):
    print(f"{word}: {frequency}")


