import requests
import json
from bayes_opt import BayesianOptimization
from difflib import SequenceMatcher
import time
from tenacity import retry, stop_after_attempt, wait_fixed
import math  # Für die Rundung von top_k

# API-Endpunkt und Header
api_url = "http://localhost:11434/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
}

# Platzhalter für die Prompts
system_prompt = """[Ihr System-Prompt hier]"""
input_prompt = """[Ihr Eingabe-Prompt hier]"""
desired_output = """[Ihre gewünschte Ausgabe hier]"""

# Funktion zur Berechnung der Ähnlichkeit zwischen zwei Texten
def compute_similarity(a, b):
    """Berechnet die Ähnlichkeit zwischen zwei Texten."""
    return SequenceMatcher(None, a, b).ratio()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def make_request(data):
    response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=60)
    response.raise_for_status()
    return response.json()

# Funktion, die die Ausgabe mit der gewünschten Antwort vergleicht und eine Ähnlichkeitsmetrik zurückgibt
def evaluate_model(temperature, top_p, frequency_penalty, presence_penalty, top_k):
    """Evaluiert das Modell mit den gegebenen Parametern und berechnet die Ähnlichkeit zur gewünschten Ausgabe."""
    # Begrenzung der Parameter auf die erlaubten Bereiche
    temperature = max(min(temperature, 0.9), 0.1)
    top_p = max(min(top_p, 0.9), 0.1)
    frequency_penalty = max(min(frequency_penalty, 1.0), -1.0)
    presence_penalty = max(min(presence_penalty, 1.0), -1.0)
    top_k = int(max(min(top_k, 100), 1))  # top_k muss eine ganze Zahl sein

    data = {
        "model": "EVA-GPT-Version6.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ],
        "max_tokens": 700,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_k": top_k  # Neu hinzugefügt
    }

    try:
        start_time = time.time()
        response_data = make_request(data)
        end_time = time.time()
        print(f"Antwortzeit: {end_time - start_time} Sekunden")

        if 'choices' in response_data and response_data['choices']:
            output = response_data['choices'][0]['message']['content'].strip()
            if not output:
                print("Die Ausgabe ist leer.")
                return 0
            # Berechnung der Ähnlichkeit mit der gewünschten Ausgabe
            similarity = compute_similarity(output, desired_output)
            print(f"Parameter: temperature={temperature}, top_p={top_p}, frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}, top_k={top_k}")
            print(f"Ausgabe: {output}\nÄhnlichkeit: {similarity}\n")
            return similarity
        else:
            print("Die Antwort enthält keine gültigen 'choices'.")
            print(f"API-Antwort: {response_data}")
            return 0
    except Exception as e:
        print(f"Fehler bei der Anfrage: {e}")
        return 0

# Definieren Sie den angepassten Parameterbereich für die Optimierung
pbounds = {
    'temperature': (0.1, 0.9),
    'top_p': (0.1, 0.9),
    'frequency_penalty': (-1.0, 1.0),
    'presence_penalty': (-1.0, 1.0),
    'top_k': (1, 100)  # Neu hinzugefügt
}

# Initialisieren Sie den Optimierer
optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

# Starten Sie die Optimierung
optimizer.maximize(
    init_points=5,
    n_iter=25,
)

# Nach Abschluss der Optimierung die besten Parameter ausgeben
print("Beste gefundene Parameter:")
print(optimizer.max)
