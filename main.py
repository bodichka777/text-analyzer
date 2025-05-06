import nltk
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import string

nltk.download('punkt_tab')
nltk.download('vader_lexicon')

file_path = "text.txt"
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"Файл '{file_path}' не знайдено. Перевір назву або шлях.")
    exit()


words = word_tokenize(text.lower())
words_clean = [word for word in words if word.isalpha()]
sentences = sent_tokenize(text)

num_words = len(words_clean)
num_sentences = len(sentences)
num_letters = sum(len(word) for word in words_clean)

print(f"Кількість слів: {num_words}")
print(f"Кількість речень: {num_sentences}")
print(f"Кількість літер: {num_letters}")


word_freq = Counter(words_clean)
most_common = word_freq.most_common(10)
print("\nНайчастіші слова:")
for word, freq in most_common:
    print(f"{word}: {freq}")

wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Хмара слів')
plt.show()

top_words = dict(most_common)
plt.figure(figsize=(10, 5))
plt.bar(top_words.keys(), top_words.values())
plt.title('Найчастіші слова')
plt.xlabel('Слово')
plt.ylabel('Кількість')
plt.show()

sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text)

print("\nАналіз настрою тексту:")
for k, v in sentiment.items():
    print(f"{k}: {v:.3f}")

compound = sentiment["compound"]

if compound >= 0.05:
    final_sentiment = "Positive 😊"
elif compound <= -0.05:
    final_sentiment = "Negative 😠"
else:
    final_sentiment = "Neutral 😐"

print(f"\nЗагальна оцінка настрою: {final_sentiment}")
