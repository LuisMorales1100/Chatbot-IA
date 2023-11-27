import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization
import random
import joblib
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE

with open("intents_Price.json",mode="r",encoding="utf-8-sig") as text:
    text = json.load(text)["intents"]

with open("intents_First_Kid.json",mode="r",encoding="utf-8-sig") as text:
    text = json.load(text)["intents"]

response_list = []
for row in range(len(text)):
    for i in range(len(text[row]["patterns"])):
        for j in range(len(text[row]["responses"])):
            t = (text[row]["tag"],text[row]["patterns"][i],text[row]["responses"][j])
            response_list.append(t)

df = pd.DataFrame(response_list, columns=["tag","patterns", "responses"])
df["patterns"] = df["patterns"].apply(lambda x: x.lower())

# Responses
df_dummies = pd.get_dummies(df["tag"],drop_first=False,prefix="tag")
df = pd.concat([df,df_dummies],axis=1)

# Tags
df_tag = df.drop_duplicates(subset=["patterns"])
df_tag = df.copy()
df_tag

count_vectorizer = feature_extraction.text.CountVectorizer()
label_encoder = LabelEncoder()

x_train = df.drop(["responses","tag","patterns"],axis=1)
x_train = df_tag.drop(["responses","tag"],axis=1)

y = label_encoder.fit_transform(df['responses'])
y = label_encoder.fit_transform(df_tag['tag'])
x_train = count_vectorizer.fit_transform(df_tag["patterns"])

encoded_labels = label_encoder.fit_transform(df['responses'])
encoded_labels = label_encoder.fit_transform(df_tag['tag'])
label_mapping = dict(zip(encoded_labels, df['responses']))
label_mapping = dict(zip(encoded_labels, df_tag['tag']))

ROS = RandomOverSampler()
X_train, X_test, y_train, y_test = train_test_split(x_train, y, test_size=0.2, random_state=42)
X_train,y_train = ROS.fit_resample(X_train,y_train)

len(y)
len(np.unique(y))
# Logistic Regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

label_mapping[predictions[0]]

count_vectorizer.inverse_transform(X_test)[0]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

scores = model_selection.cross_val_score(model, X_train,y_train, cv=10)
scores

# ANN
#tf.config.run_functions_eagerly(True)
vocab_size = 10000  # Example vocabulary size
input_dim = vocab_size + 1
len(X_train.toarray())
# Create a sequential model
model = Sequential()
# Add an Embedding layer, input_dim should be vocab_size + 1, output_dim is the size of the embedding vector
# input_length is the length of your input sequences
model.add(Embedding(input_dim=input_dim, output_dim=101, input_length=X_train.shape[1]))
# Flatten the 3D embedding tensor to 2D
model.add(Flatten())
# Add one or more Dense layers for further processing
model.add(Dense(801, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(501, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(501, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Output layer for classification

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now you can train your model using X_train and y_train
hist = model.fit(X_train.toarray(), y_train, epochs=500, batch_size=25, validation_split=0.2,verbose=1)

loss, accuracy = model.evaluate(X_train.toarray(),y_train, verbose=False)
loss_t, accuracy_t = model.evaluate(X_test.toarray(),y_test, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Test Accuracy: {:.4f}".format(accuracy_t))

pd.DataFrame(hist.history).plot(figsize=(8,5));plt.show()

confusion_matrix(y_train,np.argmax(model.predict(X_train.toarray()),axis=1))
confusion_matrix(y_test,np.argmax(model.predict(X_test.toarray()),axis=1))


# Keras Tunning
def model_builder(hp):
    hp_activation = hp.Choice('activation', values=['relu', 'tanh',"elu"])
    hp_layer_0 = hp.Int('layer_0', min_value=1, max_value=1000, step=100)
    hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
    hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
    hp_layer_3 = hp.Int('layer_3', min_value=1, max_value=1000, step=100)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model = Sequential()
    model.add(Embedding(input_dim=input_dim,output_dim=hp_layer_0, input_length=X_train.shape[1]))
    model.add(Flatten())
    model.add(Dense(units=hp_layer_1, activation=hp_activation))
    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(Dense(units=hp_layer_3, activation=hp_activation))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(model_builder,objective='val_accuracy',max_epochs=30,factor=3,directory='Models_Tunning',project_name='Tunning_1')

#stop_early = EarlyStopping(monitor='val_loss', patience=3)
tuner.search(X_train.toarray(), y_train, epochs=150, validation_split=0.2) # callbacks=[stop_early]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps.values
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train.toarray(), y_train, epochs=200, validation_split=0.2)

pd.DataFrame(history.history).plot(figsize=(8,5));plt.show()

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_train.toarray(), y_train)
loss_t, accuracy_t = model.evaluate(Models.X_Test, Models.Y_Test)
print(f"Train Loss: {loss:0.4}, Train Accuracy: {accuracy:0.4}")
print(f"Test Loss: {loss_t:0.4}, Test Accuracy: {accuracy_t:0.4}")

confusion_matrix(Models.Y_Train,np.argmax(model.predict(Models.X_Train),axis=1))
confusion_matrix(Models.Y_Test,np.argmax(model.predict(Models.X_Test),axis=1))



# Responses
while True:
    print("User: " ,end="")
    inp = input()
    if inp.lower() == "quit":
        break
    res = label_mapping[np.argmax(model.predict(count_vectorizer.transform([inp]).toarray(),verbose=0),axis=1)[0]]
    print("El Pendejo del ChatGPT: " ,end="")
    res

# Tag
while True:
    print("User: " ,end="")
    inp = input()
    if inp.lower() == "quit":
        break
    tag = label_mapping[np.argmax(model.predict(count_vectorizer.transform([inp]).toarray(),verbose=0),axis=1)[0]]
    res = random.choice(df.query(f"tag == '{tag}'")["responses"].unique())
    print("El Pendejo del ChatGPT: " ,end="")
    res

df.to_csv("Responses.csv",encoding="utf-8-sig",index=False)
model.save("Model_1.h5")
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')

converted_dict = {str(key): value for key, value in label_mapping.items()}

with open("Label_Mapping.json", 'w') as json_file:
    json.dump(converted_dict, json_file)

label_mapping
inp = input()

np.argmax(model.predict(count_vectorizer.transform([inp]).toarray()),axis=1)
label_mapping[np.argmax(model.predict(count_vectorizer.transform([inp]).toarray()),axis=1)[0]]

model.predict(X_test.toarray())
predictions = np.argmax(model.predict(X_test.toarray()),axis=1)

count_vectorizer.inverse_transform(X_test)[2]
label_mapping[predictions[2]]




model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[0]), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(len(y_train), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(tf.sparse.to_dense(X_train), y_train, epochs=100, batch_size=5, verbose=1)

loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y), verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

pd.DataFrame(hist.history).plot(figsize=(8,5))
plt.show()



## let's get counts for the first 5 tweets in the data
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())

train_vectors_Y = count_vectorizer.fit_transform(Y_train["responses"])
train_vectors_X = count_vectorizer.fit_transform(df["patterns"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(df["responses"])
clf = linear_model.LinearRegression()

scores = model_selection.cross_val_score(clf, train_vectors_X,train_vectors_Y, cv=3)
scores

clf.fit(train_vectors, train_df["target"])
sample_submission["target"] = clf.predict(test_vectors)

tfidf_vectorizer = TfidfVectorizer()

patterns_tfidf = tfidf_vectorizer.fit_transform(df['patterns']).toarray()
patterns_df = pd.DataFrame(patterns_tfidf, columns=tfidf_vectorizer.get_feature_names_out())



X_train = pd.concat([X_train,patterns_df],axis=1)





count_vectorizer = feature_extraction.text.CountVectorizer()
Responses = pd.read_csv("Responses.csv",encoding="utf-8-sig")
count_vectorizer.fit_transform(Responses["patterns"])

# Label Encoding
with open("Label_Mapping.json") as lm:
    Label_Mapping = json.load(lm)

# Model
model = tf.keras.models.load_model("Model_1.h5")

Text = {"message":"Hola"}
Text.get("message")
Tag = Label_Mapping[str(np.argmax(model.predict(count_vectorizer.transform([Text.get("message")]).toarray(),verbose=0),axis=1)[0])]
Response = random.choice(Responses.query(f"tag == '{Tag}'")["responses"].unique())









# Download NLTK resources
nltk.download('punkt')

# Keywords for intent recognition
travel_keywords = ['travel', 'trip', 'vacation', 'holiday', 'tour']


words = word_tokenize("i want to travel to cancun")
if any(word in words for word in travel_keywords):
    print("Sure, I can help you plan your trip!")
else:
    print("I'm not sure how to help with that.")