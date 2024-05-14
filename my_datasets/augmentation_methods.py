import random
import nltk
import numpy as np
nltk.download('words')
from nltk.corpus import words

def misspell_word(word):
    action = random.choice(['delete', 'repeat']) 
    letters = list(word)
    if len(letters) > 1:
        if action == 'delete':
            del letters[random.choice(range(len(letters)))]
        elif action == 'repeat':
            repeat_index = random.choice(range(len(letters)))
            letters.insert(repeat_index, letters[repeat_index])
    return ''.join(letters)

def misspell_text(text, proportion=0.5):
    words = text.split()
    num_to_misspell = int(len(words) * proportion)
    indices_to_misspell = random.sample(range(len(words)), num_to_misspell)
    for i in indices_to_misspell:
        words[i] = misspell_word(words[i])
    return ' '.join(words)

def generate_random_text():
    random_words = random.sample(words.words(), random.choice(np.arange(280, 350)))
    random_text = ' '.join(random_words)
    return random_text