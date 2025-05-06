from flask import Flask, request, jsonify, render_template
from chatbot import retrieve_answer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    response = retrieve_answer(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)