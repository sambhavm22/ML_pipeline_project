from crypt import methods
from flask import Flask
from src.logger import logging

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    
    logging.info('we are testing our second methods of logging')
    
    return "Welcome to Engineering wala bhaiyaa"


if __name__ == "__main__":
    app.run(debug=True)