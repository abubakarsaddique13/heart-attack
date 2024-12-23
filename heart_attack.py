import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit App
st.title("Heart Attack Prediction and Analysis")

# Sidebar Navigation
st.sidebar.title("Modules")
menu = st.sidebar.radio("Go to :", ["Analysis", "Model Prediction"])

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    if menu == "Analysis":
        st.write("### Exploratory Data Analysis")
        st.write("### Dataset Preview")
        st.write(df.head())


        # Data Overview
        st.write("### Data Information")
        st.write(df.info())
        st.write("statistical data analysis")
        st.write(df.describe())
        st.write("Shape of the dataset:")
        st.write("Total Rows :", df.shape[0])
        st.write("Total Columns :", df.shape[1])

        # Graph Selection
        st.write("### Select Graph Type")
        graph_type = st.selectbox("Choose a graph type", ["Box Plot", "Scatter Plot", "Correlation Heatmap"])

    
        if graph_type == "Box Plot":
            st.write("### Box Plot")
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            selected_column = st.selectbox("Select a numerical column", numerical_columns)
            if selected_column:
                fig, ax = plt.subplots()
                sns.boxplot(y=df[selected_column], ax=ax, color="lightgreen")
                ax.set_title(f"Box Plot for {selected_column}")
                st.pyplot(fig)

        elif graph_type == "Scatter Plot":
            st.write("### Scatter Plot")
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            x_axis = st.selectbox("Select X-axis", numerical_columns)
            y_axis = st.selectbox("Select Y-axis", numerical_columns)
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax, color="orange")
                ax.set_title(f"Scatter Plot between {x_axis} and {y_axis}")
                st.pyplot(fig)

        elif graph_type == "Correlation Heatmap":
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        

    elif menu == "Model Prediction":
        # Show new sidebar for model selection
        st.sidebar.write("### Select Algorithm")
        model_name = st.sidebar.selectbox("Choose a model", [
            "Logistic Regression",
            "Naive Bayes",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ])

        # Preprocessing
        st.write("### Preprocessing")
        df = df.rename(columns={
            "age": "Age",
            "sex": "Sex",
            "cp": "ChestPain",
            "trtbps": "RestingBP",
            "chol": "Cholesterol",
            "fbs": "FastingBS",
            "restecg": "RestingECG",
            "thalachh": "MaxHR",
            "exng": "ExerciseAngina",
            "oldpeak": "OldPeak",
            "slp": "ST_Slope",
            "caa": "CA",
            "thall": "Thal",
            "output": "target"
        })
        # st.write("Renamed Columns:", df.columns.tolist())

        # Splitting data
        X = df.drop("target", axis=1)
        y = df["target"]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("Data split into training and testing sets.")

        if st.sidebar.button("Train and Evaluate Model"):
            # Model Initialization
            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            elif model_name == "Support Vector Machine":
                model = SVC()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier()
            elif model_name == "XGBoost":
                model = XGBClassifier()

            # Training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            st.write(f"### {model_name} Performance")
            st.write(f"Accuracy: {acc:.2f}")
            st.write("Confusion Matrix:")
            st.write(cm)
            st.write("Classification Report:")
            st.dataframe(report_df.style.format(precision=2))
