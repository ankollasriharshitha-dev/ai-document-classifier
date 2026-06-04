from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = 0

    if request.method == "POST":

        text = request.form.get("text", "")

        uploaded_file = request.files.get("document")

        if uploaded_file and uploaded_file.filename:
            text = uploaded_file.read().decode("utf-8")

        if text.strip():

            transformed_text = vectorizer.transform([text])

            prediction = model.predict(transformed_text)[0]

            confidence = max(
                model.predict_proba(transformed_text)[0]
            ) * 100

            confidence = round(confidence, 2)

            if confidence < 50:
                prediction = "Unknown Category"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)