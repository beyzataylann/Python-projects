from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from sentence_transformers import SentenceTransformer
import numpy as np

def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = [line.strip() for line in f if line.strip()]
    return stopwords


turkce_kelimeler = [
    "mekan", "mekanik", "mektep", "emek", "çekmek", "kitap",
    "kalem", "kalın", "araba", "armut", "asker", "aslan",
    "bilgi", "bilim", "biber", "bilgisayar", "makine", "yazılım", "donanım",
    "koşmak", "gitmek", "gelmek", "görmek", "konuşmak", "yemek", "içmek",
    "güzel", "hızlı", "büyük", "küçük", "soğuk", "sıcak", "uzun", "kısa",
    "mutlu", "üzgün", "zor", "kolay"
]

stopwords_from_file = load_stopwords("data/stopwords_tr.txt")  

all_keywords = list(set(turkce_kelimeler + stopwords_from_file))


model = SentenceTransformer("dbmdz/bert-base-turkish-cased")
turkce_stopwords = set(stopwords_from_file)
embeddings = model.encode(all_keywords).astype('float32')

class FaissAutocomplete(Completer):
    def __init__(self, keywords, stopwords, model):
        self.keywords = keywords
        self.stopwords = stopwords
        self.model = model

    def get_completions(self, document, complete_event):
        text = document.text.strip()
        if not text or len(text) < 2:
            return

        candidates = [w for w in self.keywords if w.startswith(text)]
        if not candidates:
            return

        embeddings_candidates = self.model.encode(candidates).astype('float32')
        vector = self.model.encode([text]).astype('float32')

        dists = np.linalg.norm(embeddings_candidates - vector, axis=1)
        nearest_indices = np.argsort(dists)[:5]

        for idx in nearest_indices:
            suggestion = candidates[idx]
            if suggestion in self.stopwords:
                continue
            yield Completion(suggestion, start_position=-len(text))

faiss_completer = FaissAutocomplete(all_keywords, turkce_stopwords, model)

while True:
    try:
        user_input = prompt("Kelime girin (çıkmak için q): ", completer=faiss_completer)
        if user_input.lower() == "q":
            break
        print("Seçilen:", user_input)
    except KeyboardInterrupt:
        break
