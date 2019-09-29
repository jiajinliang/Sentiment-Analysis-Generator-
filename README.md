# Review Sentiment Analyzer and Fake News Detector

To run, git clone and open the Directory in Windows Command Prompt (cmd.exe), then run `py -3 Sentiment_Analysis.py` or `py -3 FakeNews_Detector.py`. But before it finally works, you will need to install Python 3 for Windows and the required Python packages based on the error messages when you try running. Two of the required packages are given as wkhtmltoimage.exe and wkhtmltopdf.exe, simply run them to install these two packages.

*** Files ***

- ./*/sentiment.py
	This file is the high level interface of the data and sentiment class.
	It contains the inplementations of functions that process the data and
	sentiment building.

- ./*/classify.py
	This file is the interface of the logistic regression classifier. It
	contains two functions that can build the model and evaluate the model.

- ./sup/Supervised.ipynb
	This file contains the supervised classifier building process including
	parameter searching, feature engineering, and evaluations.

- ./sup/Semi-Supervised.ipynb
	This file contains the semi supervised classifier building process
	including parameter searching, feature engineering, and evaluations.

- ./*/all_in_one.py
	This file integrates several files in one and contains all the codes needed
	to train the logistic regression model from the data for the review sentiment
	analysis with the parameters and vectorization methods that maximize the
	accuracy. `Sentiment_Analysis.py` will call methods from this file and display
	the outputs to the PyQt5 application

- ./*/all_in_one_P2.py
	This file is similar with `all_in_one.py` but the data and the model are for
	Fake News Detection, `FakeNews_Detector.py` will call methods from this file
	and display outputs to the PyQt5 application