from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

JOKE_API_URL = "https://v2.jokeapi.dev/joke/Any"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_joke")
def get_joke():
    response = requests.get(JOKE_API_URL)
    if response.status_code == 200:
        joke_data = response.json()
        if joke_data["type"] == "single":
            joke = joke_data["joke"]
        else:
            joke = f"{joke_data['setup']} - {joke_data['delivery']}"
        return jsonify({"joke": joke})
    return jsonify({"joke": "Oops! Couldn't fetch a joke at the moment."})

if __name__ == "__main__":
    app.run(debug=True)
