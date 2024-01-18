# ML-Projekt

Zum Einlesen der Daten im Format 256x128 gibt es eine read_data Funktion in read_data.py. Nutzung:
    import read_data
    data = read_data.read_data(300)
Der Parameter in read_data ist die maximale Anzahl an Spektrogrammen pro Klasse. Übersteigt er die Anzahl der Spektrogramme pro Klasse so werden nur die existierenden
zurückgegeben. Im Dataframe data liegen nun die Spektrogramme (data['arr]) und Targets (data['target]) sowie der Name der Klasse, die Nummer des Calls, das DB-Format 
und eine ID.

In pca.ipynb werden die Daten mehrere Kompressionsräume transformiert. Anschließend visualisiert und ML-Verfahren darauf angewendet.


