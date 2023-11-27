from flask import Flask, jsonify, request, render_template
import mysql.connector
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import random
import joblib

server = "sql5.freesqldatabase.com"
database = "sql5664042"
username = "sql5664042"
password = "mwQUwFLr4D"
port = "3306"

app = Flask(__name__)

#count_vectorizer = feature_extraction.text.CountVectorizer()
Responses = pd.read_csv("Responses.csv",encoding="utf-8-sig")
count_vectorizer = joblib.load("count_vectorizer.pkl")
#count_vectorizer.fit_transform(Responses["patterns"])

# Label Encoding
with open("Label_Mapping.json") as lm:
    Label_Mapping = json.load(lm)

# Model
model = tf.keras.models.load_model("Model_1.h5")

@app.route("/",methods=["GET"])
def index():
    return render_template("base.html")

@app.route("/api/Chatbot/Predict",methods=["POST"])
def Predict():
    Text = request.get_json().get("message")
    Label = str(np.argmax(model.predict(count_vectorizer.transform([Text]).toarray(),verbose=0),axis=1)[0])
    Tag = Label_Mapping[Label]
    Response = random.choice(Responses.query(f"tag == '{Tag}'")["responses"].unique())
    try:
        connection = mysql.connector.connect(user=username,password=password,host=server,database=database)
        cursor = connection.cursor()
        cursor.execute("INSERT INTO Chatbot (Pattern,Response) VALUES(%s,%s)",(Text,Label))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({"Answer": Response})
    except:
        return jsonify({"Answer": Response})

if __name__ == "__main__":
    app.run(debug=True)