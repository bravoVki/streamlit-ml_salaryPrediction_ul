import streamlit as st  # to create app
import pickle  #to load data 
import numpy as np 

# creating a function to load model
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()  #for execution

#accessing keys
regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

#creating prediction page ie. building streamlit page
def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### WE NEED SOME INFO.. TO PREDICT THE SALARY ###""")

    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )
# select input
    
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
#slider for exp have min & max value
    expericence = st.slider("Years of Experience", 0, 50, 3)

#button to start prediction--> when clicked ok=1
    ok = st.button("Calculate Salary")

    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
#calling regressor model to predict
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")