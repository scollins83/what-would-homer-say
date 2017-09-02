from flask import Flask, render_template
import model

app = Flask(__name__)


@app.route("/")
def home():
    homer_utterance = model.infer()
    return render_template("index.html", homer_utterance=homer_utterance)

if __name__ == "__main__":
    app.run(debug=True)