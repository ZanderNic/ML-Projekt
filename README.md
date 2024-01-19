# ML-Projekt

Zum Einlesen der Daten im Format 256x128 gibt es eine read_data Funktion in read_data.py. Nutzung:
    import read_data
    data = read_data.read_data(300)
Der Parameter in read_data ist die maximale Anzahl an Spektrogrammen pro Klasse. Übersteigt er die Anzahl der Spektrogramme pro Klasse so werden nur die existierenden
zurückgegeben. Im Dataframe data liegen nun die Spektrogramme (data['arr]) und Targets (data['target]) sowie der Name der Klasse, die Nummer des Calls, das DB-Format 
und eine ID.

In pca.ipynb werden die Daten mehrere Kompressionsräume transformiert. Anschließend visualisiert und ML-Verfahren darauf angewendet.


## Übersicht über die Dateien

| Dateiname | Beschreibung |
|---|---|
| ML-Projekt.pptx | Folien zur Präsentation |
| bat23423.mp3 | Für Menschen hörbares Beispielaudio aus der Präsentation |
| pca.ipynb | Teil 1: PCA, KNN, Random Forest, AdaBoost, Gradient Boosting, einfaches FFNN |
| cnn_data_preprocessing.ipynb | Teil 3: CNN, Datenvorverarbeitung & manuelle Dimensionsreduktion, FFNN |
| read_data.py | Gemeinsamer Einlesecode für Teil 1 & Teil 2 |
| orig.keras | CNN trainiert auf den Originaldaten |
| my.keras | CNN trainiert mit unserer eigenen Vorerarbeitungsfunktion |
| ffnn.keras | FFNN trainiert mit den reduzierten Daten |
