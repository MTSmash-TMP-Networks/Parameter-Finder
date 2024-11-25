# Parameter-Finder für KI-Modelle

Dieses Projekt dient der Optimierung von Parametern für generative Sprachmodelle wie **EvaGPT-German**, **LLama** oder andere LLMs (Large Language Models). Mithilfe von **Bayesian Optimization** wird die bestmögliche Kombination der Modellparameter (`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `top_k`) ermittelt, um die Ausgabequalität zu verbessern.

## Funktionen und Aufbau

### 1. **Systemübersicht**
Das Skript kommuniziert über eine REST-API mit dem Modell und verwendet das Bayes'sche Optimierungsverfahren, um die Parameterkombination zu finden, die die Ausgabe des Modells am besten an ein vorgegebenes Ziel anpasst.

### 2. **Parameter**
- `temperature`: Steuert die Kreativität des Modells (zwischen 0.1 und 0.9).
- `top_p`: Begrenzung der Wahrscheinlichkeitssumme für die Auswahl der nächsten Tokens (zwischen 0.1 und 0.9).
- `frequency_penalty`: Reduziert die Wiederholung von Tokens (zwischen -1.0 und 1.0).
- `presence_penalty`: Fördert oder entmutigt die Verwendung neuer Tokens (zwischen -1.0 und 1.0).
- `top_k`: Begrenzung der Auswahl der nächsten Tokens auf die Top-K wahrscheinlichsten Optionen (zwischen 1 und 100).

### 3. **Ablauf**
1. **Anfrage senden**: Das Skript sendet eine Anfrage an das Modell mit einer bestimmten Kombination der Parameter.
2. **Antwort vergleichen**: Die Modellantwort wird mit einer Zielantwort verglichen.
3. **Ähnlichkeit berechnen**: Die Ähnlichkeit zwischen der Modellantwort und der Zielantwort wird mit der **SequenceMatcher**-Bibliothek berechnet.
4. **Optimierung durchführen**: **BayesianOptimization** passt die Parameter iterativ an, um die Ähnlichkeit zu maximieren und somit die Ausgabequalität zu verbessern.

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
- `tenacity`: Für robustes Retry-Handling.

### Installation
Die benötigten Python-Bibliotheken können mit folgendem Befehl installiert werden:
```bash
pip install requests bayesian-optimization tenacity
```

## Verwendung

1. **Starten Sie den API-Server**:
   Stellen Sie sicher, dass der Endpunkt (`http://localhost:11434/v1/chat/completions`) aktiv ist und das Modell bereit ist, Anfragen zu verarbeiten.

2. **Konfigurieren Sie die Prompts und Zielantwort**:
   - **System-Prompt**: Definieren Sie die Systemvorgaben für das Modell.
   - **Eingabe-Prompt**: Legen Sie die Eingabe fest, auf die das Modell reagieren soll.
   - **Zielantwort**: Geben Sie die gewünschte Antwort an, mit der die Modellantwort verglichen wird.

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

## Erweiterte Konfiguration

### Anpassung der Parameterbereiche
Die Bereiche für die Parameter können in der `pbounds`-Variable im Skript angepasst werden, um spezifische Anforderungen Ihres Modells oder Ihrer API zu erfüllen.

### Beispiel für `pbounds`:
```python
pbounds = {
    'temperature': (0.1, 0.9),
    'top_p': (0.1, 0.9),
    'frequency_penalty': (-1.0, 1.0),
    'presence_penalty': (-1.0, 1.0),
    'top_k': (1, 100)
}
```

### Hinzufügen weiterer Parameter
Falls Ihr Modell zusätzliche Parameter unterstützt, können diese ähnlich wie `top_k` hinzugefügt und in der Optimierungsfunktion berücksichtigt werden.

## Fehlerbehandlung und Logging

Das Skript verwendet die **tenacity**-Bibliothek, um fehlgeschlagene Anfragen automatisch neu zu versuchen. Für eine bessere Nachverfolgung und Debugging können Sie zusätzlich Logging-Mechanismen implementieren.

## Hinweise

- **API-Unterstützung für `top_k`:** Vergewissern Sie sich, dass Ihr Modell und die API `top_k` als Parameter unterstützen. Nicht alle Modelle oder APIs unterstützen diesen Parameter.
- **Ganzzahligkeit von `top_k`:** Da `top_k` eine ganze Zahl sein muss, wird es im Skript entsprechend umgewandelt. Stellen Sie sicher, dass dies in Ihrem Kontext akzeptabel ist.
- **Leistung und Ressourcen:** Bayesian Optimization kann ressourcenintensiv sein. Passen Sie `init_points` und `n_iter` entsprechend Ihrer verfügbaren Rechenleistung an.

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

## Beiträge

Beiträge sind willkommen! Bitte erstellen Sie einen Pull-Request oder öffnen Sie ein Issue, um Änderungen vorzuschlagen.
