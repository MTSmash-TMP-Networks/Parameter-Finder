import requests
import json
from bayes_opt import BayesianOptimization
from difflib import SequenceMatcher

# API-Endpunkt und Header
api_url = "http://localhost:11434/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
}

# System-Prompt hinzufügen
system_prompt = """
Du bist EvaGPT-German, eine leistungsstarke, datenschutzorientierte KI, die speziell für die deutsche Sprache entwickelt wurde. Dein Hauptfokus liegt darauf, Nutzern ohne Internetverbindung bestmögliche Unterstützung zu bieten. Deine Antworten sollen präzise, höflich und hilfreich sein.

Richtlinien:

    1. Datenverarbeitung: Du arbeitest ohne externe Verbindungen oder Internetzugriff. Alle Berechnungen und Antworten basieren auf lokal verfügbaren Daten und Modellen.
    2. Privatsphäre: Du respektierst die Privatsphäre der Benutzer und sammelst keine persönlichen Daten. Sensible Informationen werden niemals weitergegeben oder gespeichert.
    3. Sprache: Deine Antworten sollen in klarer, verständlicher deutscher Sprache formuliert sein. Verwende eine formale, aber zugängliche Ansprache, es sei denn, der Benutzer wünscht eine informellere Kommunikation.
    4. Kontextverstehen: Du behältst den Kontext früherer Interaktionen im Hinterkopf, um präzisere Antworten zu liefern. Sei jedoch achtsam und gebe niemals unnötige Details aus früheren Gesprächen preis, es sei denn, es ist ausdrücklich gewünscht.
    5. Zielgerichtete Unterstützung: Dein Ziel ist es, dem Benutzer bei technischen und alltäglichen Aufgaben zu helfen, komplexe Sachverhalte zu erklären und bei der Entwicklung von Software, insbesondere cloudbasierten Anwendungen und KI-Projekten, zu unterstützen.
    6. Fehlerhandling: Wenn du eine Anfrage nicht vollständig verstehst oder die Antwort außerhalb deines Wissensbereichs liegt, signalisiere dies höflich und versuche, dem Benutzer hilfreiche Alternativen oder allgemeine Informationen zu bieten.
    7. Modularität: Du bist anpassungsfähig und lernfähig, basierend auf lokal bereitgestellten Daten, um die Bedürfnisse des Benutzers bestmöglich zu erfüllen.
"""

# Eingabeprompt und gewünschte Musterantwort
input_prompt = "Geben Sie eine kurze Beschreibung von Python."
desired_output = "Python ist eine interpretierte, vielseitige und leicht zu erlernende Programmiersprache, die für ihre klare Syntax und Lesbarkeit bekannt ist. Sie unterstützt verschiedene Programmierparadigmen wie objektorientierte, prozedurale und funktionale Programmierung. Python wird häufig in Bereichen wie Webentwicklung, Datenanalyse, maschinellem Lernen, Automatisierung und wissenschaftlichem Rechnen eingesetzt. Dank einer großen Standardbibliothek und einer aktiven Community bietet Python leistungsstarke Tools für unterschiedlichste Anwendungsgebiete."

# Funktion zur Berechnung der Ähnlichkeit zwischen zwei Texten
def compute_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Funktion, die die Ausgabe mit der gewünschten Antwort vergleicht und eine Ähnlichkeitsmetrik zurückgibt
def evaluate_model(temperature, top_p, frequency_penalty, presence_penalty):
    temperature = max(min(temperature, 1), 0)  # Begrenzung auf [0,1]
    top_p = max(min(top_p, 1), 0)
    frequency_penalty = max(min(frequency_penalty, 2), -2)  # Begrenzung auf [-2,2]
    presence_penalty = max(min(presence_penalty, 2), -2)

    data = {
        "model": "EVA-GPT-Version6.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ],
        "max_tokens": 500,  # Erhöht, um längere Antworten zu ermöglichen
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        output = response_data['choices'][0]['message']['content'].strip()

        # Berechnung der Ähnlichkeit mit der gewünschten Ausgabe
        similarity = compute_similarity(output, desired_output)
        print(f"Ausgabe: {output}\nÄhnlichkeit: {similarity}\n")
        return similarity
    else:
        print(f"Fehler bei der Anfrage: {response.status_code}")
        return 0  # Schlechte Ähnlichkeit bei Fehler

# Definieren Sie den Parameterbereich für die Optimierung
pbounds = {
    'temperature': (0.0, 1.0),
    'top_p': (0.0, 1.0),
    'frequency_penalty': (-2.0, 2.0),
    'presence_penalty': (-2.0, 2.0)
}

# Initialisieren Sie den Optimierer
optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    verbose=2,  # 0 = nichts ausgeben, 1 = minimal, 2 = alles
    random_state=1,
)

# Starten Sie die Optimierung
optimizer.maximize(
    init_points=5,  # Anzahl der zufälligen Initialisierungspunkte
    n_iter=25,      # Anzahl der Optimierungsschritte
)

# Nach Abschluss der Optimierung die besten Parameter ausgeben
print("Beste gefundene Parameter:")
print(optimizer.max)
