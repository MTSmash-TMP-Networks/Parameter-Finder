# Parameter-Finder für KI-Modelle

Dieses Projekt dient der Optimierung von Parametern für generative Sprachmodelle wie **EvaGPT-German**, **LLama** oder andere LLMs (Large Language Models). Mithilfe von **Bayesian Optimization** wird die bestmögliche Kombination der Modellparameter (`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`) ermittelt, um die Ausgabequalität zu verbessern. 

## Funktionen und Aufbau

### 1. **Systemübersicht**
Das Skript kommuniziert über eine REST-API mit dem Modell und verwendet das Bayes'sche Optimierungsverfahren, um die Parameterkombination zu finden, die die Ausgabe des Modells am besten an ein vorgegebenes Ziel anpasst.

### 2. **Parameter**
- `temperature`: Steuert die Kreativität des Modells (zwischen 0 und 1).
- `top_p`: Begrenzung der Wahrscheinlichkeitssumme für die Auswahl der nächsten Tokens (zwischen 0 und 1).
- `frequency_penalty`: Reduziert die Wiederholung von Tokens (zwischen -2 und 2).
- `presence_penalty`: Fördert oder entmutigt die Verwendung neuer Tokens (zwischen -2 und 2).

### 3. **Ablauf**
- Das Skript sendet eine Anfrage an das Modell und vergleicht die Modellantwort mit einer Zielantwort.
- Die Ähnlichkeit zwischen der Modellantwort und der Zielantwort wird mit der **SequenceMatcher**-Bibliothek berechnet.
- Die Optimierung wird durch **BayesianOptimization** durchgeführt, das die Parameter iterativ anpasst, um die Ähnlichkeit zu maximieren.

### 4. **Endpunkt und API**
Das Modell wird über einen lokalen oder entfernten API-Endpunkt angesprochen. Standardmäßig ist der lokale Endpunkt:
```
http://localhost:11434/v1/chat/completions
```

Die Konfiguration des Headers:
```json
{
    "Content-Type": "application/json"
}
```

Falls ein anderer Endpunkt verwendet wird, kann die URL im Code angepasst werden.

## Anforderungen

### Python-Pakete
- `requests`: Für die Kommunikation mit der API.
- `json`: Zum Umgang mit JSON-Daten.
- `bayes_opt`: Für die Optimierung der Parameter.
- `difflib`: Zur Berechnung der Ähnlichkeit zwischen Texten.

### Installation
Die benötigten Python-Bibliotheken können mit folgendem Befehl installiert werden:
```bash
pip install requests bayesian-optimization
```

## Verwendung

1. **Starten Sie den API-Server**:
   Stellen Sie sicher, dass der Endpunkt (`http://localhost:11434/v1/chat/completions`) aktiv ist und das Modell bereit ist, Anfragen zu verarbeiten.

2. **Konfigurieren Sie die Zielantwort**:
   Aktualisieren Sie die Variable `desired_output`, um die gewünschte Antwort für die Optimierung festzulegen.

3. **Optimierung starten**:
   Führen Sie das Skript aus. Die Optimierung beginnt mit zufälligen Initialisierungspunkten und verfeinert die Parameter in mehreren Iterationen:
   ```bash
   python parameter-finder.py
   ```

4. **Ergebnisse überprüfen**:
   Nach Abschluss der Optimierung werden die besten Parameter in der Konsole ausgegeben:
   ```
   Beste gefundene Parameter:
   {'target': ..., 'params': {...}}
   ```

## Ergebnisse

Das Skript zeigt die Modellantwort, die berechnete Ähnlichkeit und die Optimierungsergebnisse. Diese können verwendet werden, um das Modell weiter zu verbessern und an spezifische Anwendungsfälle anzupassen.

## Anwendungsfälle

- **Generative Sprachmodelle:** Optimierung der Parameter für feinjustierte Antworten.
- **Experimentelles Finetuning:** Bestimmen der optimalen Parameter nach einem Finetuning-Prozess.
- **Vielfältige Modelle:** Unterstützung für beliebige Sprachmodelle, die über eine REST-API angesprochen werden können, wie **LLama**, **OpenAI GPT**, **Falcon**, **BLOOM** und mehr.
