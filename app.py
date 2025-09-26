from flask import Flask, render_template, request
import joblib

# Load model + vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news_text = request.form["news"]
        if news_text.strip():
            input_data = vectorizer.transform([news_text])
            prediction = model.predict(input_data)[0]
            result = "✅ Real News" if prediction == "REAL" else "⚠️ Fake News"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
