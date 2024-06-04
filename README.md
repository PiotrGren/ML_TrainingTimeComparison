**Choose your language / Wybierz język**

[EN](#english) / [PL](#polski)

#### English

# Analysis and optimization of various machine learning models, in the context of training time using anomaly detection as an example

## Table of Contents

1. [Description](#description)
2. [Collecting Data](#collecting-data)
3. [Creating DataCollector in Windows](#creating-datacollector-in-windows)
4. [License](#license)
5. [Authors](#authors)

## Description

This project aims to analyze and optimize various machine learning models, such as Isolation Forest, K-Means, Logistic Regression and Local Outlier Factor, in the context of model training time. The code first trains the models on the collected data and then runs tests to compare their performance in different configurations. The results are visualized with graphs that show the times or accuracy of each model's performance. In addition, the project includes an example of exporting a trained model using the `joblib` library for later use.

## Collecting Data

A DataCollector in the Performance Monitor was created to collect CPU usage data from the Resource Monitor. We created a new DataCollector by selecting and configured it, choosing “Performance Counter” and Processor as the data source. “% CPU utilization” was selected as the parameter for data collection. The DataCollector was configured to collect data to a CSV file.

## Creating DataCollector in Windows.

1 Open the "Resource Monitor" program available in Windows.
2. Go to the "Data Collector Sets" tab and select the “User Defined” tab.
3. Create a new DataCollector by selecting the option to create it manually.
4. Select "Performance Counter" and Processor as the data source.
5. Select "% CPU utilization" as the parameter for data collection.
6. Configure the DataCollector to save the data to a CSV file in a folder of your choice.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Authors

Greń Piotr - Co-developer - https://github.com/PiotrGren

Kiwacka Gabriela - Co-developer - https://github.com/GabrielaKiwacka


#### Polski

# Analiza i optymalizacja różnych modeli uczenia maszynowego, w kontekście czasu szkolenia na przykładzie wykrywania anomalii

## Spis treści

1. [Opis](#opis)
2. [Zebranie Danych](#zebranie-danych)
3. [Tworzenie DataCollector w systemie Windwos](#tworzenie-datacollector-w-systemie-windows)
4. [Licencja](#licencja)
5. [Autorzy](#autorzy)

## Opis

Projekt ten ma na celu analizę i optymalizację różnych modeli uczenia maszynowego, takich jak Isolation Forest, K-Means, Logistic Regression i Local Outlier Factor, w kontekście czasu trenowania modelu. Kod najpierw trenuje modele na zebranych danych, a następnie przeprowadza testy, aby porównać ich wydajność w różnych konfiguracjach. Wyniki są wizualizowane za pomocą wykresów, które przedstawiają czasy lub dokładność działania poszczególnych modeli. Dodatkowo, projekt zawiera przykład wyeksportowania wyszkolonego modelu przy użyciu biblioteki `joblib`, co umożliwia ich późniejsze wykorzystanie.

## Zebranie danych

W celu zebrania danych dotyczących zużycia procesora został stworzony DataCollector w monitorze wydajności, który zbiera informacje z monitora zasobów. Utworzyliśmy nowy DataCollector, wybierając i skonfigurowaliśmy go, wybierając „Licznik wydajności” oraz Procesor jako źródło danych. Jako parametr do zbierania danych wybrany został parametr „%wykorzystanie procesora”. DataCollector został skonfigurowany tak, aby zbierał dane do pliku CSV.

## Tworzenie DataCollector w systemie Windows

1. Otwórz program „Resource Monitor” dostępny w systemie Windows.
2. Przejdź do zakładki „Zestawy modułów zbierających dane” (ang. Data Collector Sets) i wybierz zakładkę „Zdefiniowany przez użytkownika”.
3. Stwórz nowy DataCollector, wybierając opcję stworzenia go ręcznie.
4. Wybierz „Licznik wydajności” oraz Procesor jako źródło danych.
5. Jako parametr do zbierania danych wybierz „%wykorzystanie procesora”.
6. Skonfiguruj DataCollector tak, aby zapisywał dane do pliku CSV w wybranym przez Ciebie folderze.

## Licencja

Ten projekt jest licencjonowany na licencji MIT. Zobacz plik LICENSE aby uzyskać więcej informacji.

## Autorzy

Greń Piotr - Współtwórca - https://github.com/PiotrGren

Kiwacka Gabriela - Współtwórca - https://github.com/GabrielaKiwacka
