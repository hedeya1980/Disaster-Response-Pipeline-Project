# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Enter `http://127.0.0.1:3000` in the address bar in a browser

## Dependencies for Running Locally

* attrs==21.2.0
* certifi==2022.6.15
* click==8.1.3
* decorator==5.1.0
* Flask==2.3.2
* idna==3.3
* importlib-metadata==4.0.1
* ipython-genutils==0.2.0
* itsdangerous==2.1.2
* Jinja2==3.1.2
* jsonschema==4.1.0
* jupyter-core==4.8.1
* MarkupSafe==2.1.2
* nltk==3.8.1
* nbformat==5.1.3
* numpy @ file:///D:/Downloads/numpy-1.21.2%2Bmkl-cp310-cp310-win_amd64.whl
* pandas==2.0.0
* plotly==5.13.1
* pyrsistent==0.18.0
* python-dateutil==2.8.2
* pytz==2021.3
* requests==2.28.1
* six==1.16.0
* sklearn==0.0
* traitlets==5.1.0
* typing_extensions==4.5.0
* urllib3==1.26.12
* Werkzeug==2.3.4
