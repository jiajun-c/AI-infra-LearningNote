import nltk
from nltk import bigrams, trigrams
from nltk.corpus import reuters
from collections import defaultdict
print(nltk.find("."))
# Download necessary NLTK resources
nltk.download('reuters')
nltk.download('punkt')
words = nltk.word_tokenize(' '.join(reuters.words()))

tri_gram = list(trigrams(words))

model = defaultdict(lambda: defaultdict(lambda: 0))

for w1, w2, w3 in tri_gram:
    model[(w1, w2)][w3] += 1

for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

def predict_next_word(w1, w2):
    """
    Predicts the next word based on the previous two words using the trained trigram model.
    Args:
    w1 (str): The first word.
    w2 (str): The second word.

    Returns:
    str: The predicted next word.
    """
    next_word = model[w1, w2]
    if next_word:
        predicted_word = max(next_word, key=next_word.get)  # Choose the most likely next word
        return predicted_word
    else:
        return "No prediction available"

# Example usage
print("Next Word:", predict_next_word('the', 'stock'))