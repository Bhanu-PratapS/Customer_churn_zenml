from flask import Flask
from src.logger import logger

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    logger.info("This is an info message")
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)
