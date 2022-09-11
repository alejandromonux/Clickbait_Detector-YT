# Clickbait_Detector-YT
###### Final University Grade Project on Clickbait Detection with NLP, focused on YT titles

# Description

This project, meant to serve as my Final Thesis for the Computer Engineering college degree, encompasses all the experimentations and process of development I've gone through to create a prediction model with the aim to detect clickbait in youtube through the use of different NLP techniques. The model has the structure of the following image:

![Model Diagram](https://i.imgur.com/ilxHwQS.png)

This project is meant to be used as a server with the model hosted inside it or as a testing ground by using just the python code of it, accessible from the main.py file in the DataRetrieval directory.
# Directory Structure
- app
  - templates
- Classification
  - BERT
  - Boosting
  - Clustering
  - DecisionTree
  - LogisticRegression
  - Model
  - NaiveBates
  - RandomForest
  - SentimentAnalysis
- DataRetrieval
- Encoding
- Tools

# Databases
All databases and .json are located in "DataRetrieval". The most important ones are:
> "adjusted_database.json". A complete revised database of examples.
> "encoded_database.json". An encoded version of adjusted_database.json. Will be fed to the model.
> "channels.json". Contains the information for the retrieval of data from the Youtube API.

We also have, in the same directory, a series of file which store the trained pieces of the prediction model, namely:
> "RandomForest.json". It stores the data of the trained object of a RandomForest model.
> "XGBoost.json". It stores the data of the trained object of an XGBoost model.
> "UnionLayer.json". It stores the data of the trained object of a MultiLayerPerceptron Model.

# Requirements
- Python version 3.7.
- All package requirements are listed in "requirements.txt".
- Any GPU for the final model to run correctly.
- 20 GB of space.
# Execution
 
### Execution of the development section
The main file used for this matter is the main.ph located inside the DataRetrieval folder. The structure of the executable parameters is:
> option1 option2 option3
- Option1: This parameter selects the mode in which the program will run. The different modes are listed in an if-else structure in main.py.
- Option2: This parameter selects a subsection of said modes. Also listed in main.py in the same if-else.
- Option3: This parameter serves the purpose of choosing between some options in this main.py, namely the selection of one database or the other or the selection of one model over the rest, for example.

### Execution of the web server section
If we want to execute the server, all the necessary files for it are locates in the "app" directory. There is all the server programming.
In order to get the server running, we need waitress so we can start a WSGI server. The command meant for this task is:
> waitress-serve --listen=0.0.0.0:<port> --url-scheme=https app:app    

Once we've done this we've got a server running on localhost. and listening to the port we introduced, but this server is not yet accessible to the whole world.
If you want to make it so it is, the option I've chosen during the development period of the project is to use ngrok. You can use any hosting service for that task.
The command to use in order to launch ngrok in windows is:
> ./ngrok.exe http <port>