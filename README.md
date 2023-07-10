# msc-data-science-project-2020_21---files-ehine02
msc-data-science-project-2020_21---files-ehine02 created by GitHub Classroom

![Overview Diagram](module_data_flow.jpg?raw=true "Modules and Data Flow")

### To download the StatsBomb data
from sb_interface import store_events, preprocess_events

store_events()

preprocess_events()


### To run a numeric sequence classification, for example;
from numeric_sequence import classification

result = classification(sample_size=1000, epochs=10)

### To run a regression;
result = regression(sample_size=1000, epochs=10)

## Index of files with brief description to aid orientation.

[sb_interface.py](sb_interface.py) - data extraction, pre-processing and storage via the StatsBomb API

[numeric_sequence.py](numeric_sequence.py) - implementation of numeric sequence modelling approach and the TF2/Keras models for Chance classification and xG regression

[text_sequence.py](text_sequence.py) - implementation of text sequence modelling approach and the TF2/Keras models for Chance classification and xG regression

[xg_utils.py](xg_utils.py) - implementation of the xG model and map utility

[viz.py](viz.py) - provision for the various visualisations

[test_runner.py](test_runner.py) - test harness for executing multiple tests of different approaches and models

[utils.py](utils.py) - miscellaneous helper functions

[lr_agg.py](lr_agg.py) - basic implementation of Logistic Regression using scikit-learn, including method for aggregation of sequence data

[wame_opt.py](wame_opt.py) - WAME training optimiser, borrowed as implemented for ML module coursework

[attributes.py](attributes.py) - defines list of relevant attributes to extract/retain from StatsBomb

[test_utils.py](test_utils.py) - basic unit tests for functions in utils.py
