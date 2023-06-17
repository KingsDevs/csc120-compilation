from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', title="CSC 120 Compilation")

@app.route('/rl')
def serve_webgl():
    return render_template("rl-game.html")


if __name__ == '__main__':
    app.run(debug=True)

