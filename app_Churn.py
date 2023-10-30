
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


# Load the model and encoder
with open('model_churn.pkl', 'rb') as file:
    model_churn, gender_encoder = pickle.load(file)

# Define the Streamlit app
st.title("Churn Prediction App")

# Create two tabs: "Predict Churn" and "Predict from CSV"


tab1, tab2 = st.tabs( ["📈 Predict Churn", " 🗃 Predict from CSV"])

with tab1:
    st.header(" 📈 Predict Churn")
    age = st.slider('Age:', 0, 100, 50)
    #age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_purchase_amount = st.number_input("Total Purchase Amount")

    if st.button("Predict"):
        # Create a DataFrame for prediction
        x_new = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'total_purchase_amount': [total_purchase_amount]
        })

        # Encode gender
        x_new['gender'] = gender_encoder.transform(x_new['gender'])

        # Predict churn
        y_pred_new = model_churn.predict(x_new)

        # Convert prediction to result
        result = "Churned" if y_pred_new[0] == 1 else "Retained"

        
        st.markdown(f"Prediction result :  <span style='color:red' 'font-size: 250%'> {result}</span>",
             unsafe_allow_html=True)

with tab2:
    st.header(" 🗃 Predict from CSV")
    st.caption('*** File detail with [age, gender, total_purchase_amount]')
    csv_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
    

    if csv_file is not None:
        df = pd.read_csv(csv_file)

        # Encode gender in the CSV data
        df['gender'] = gender_encoder.transform(df['gender'])

        # Predict churn for the entire CSV
        y_pred_csv = model_churn.predict(df)

        # Convert predictions to results
        results = np.where(y_pred_csv == 0, 'Retained', 'Churned')
        df['Predict Churn'] = results
        df['gender'] = np.where(df['gender'] == 1, 'Male', 'Female')
        st.write("Predicted results:")
        st.write(df)

        # if st.button("Download Predictions as CSV"):
        st.download_button("Download CSV", df.to_csv(index=False), file_name="churn_predictions.csv")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
        for ax, multiple in zip((ax1, ax2), ['layer', 'fill']):
                sns.histplot(data=df, x='gender', hue='Predict Churn', binwidth=10, stat='percent', multiple=multiple, ax=ax)
                ax.set_title(f"multiple='{multiple}'")
            # ax.bar_label(ax.containers[0], fmt='%.2f',label_type='center', color='black')
            # ax.bar_label(ax.containers[1], fmt='%.2f', label_type='center', color='white')
        for bar_group, color in zip(ax.containers, ['black', 'white']):
                ax.bar_label(bar_group, label_type='center', color=color,
                            labels=[f'{bar.get_height() * 100:.1f} %' if bar.get_height() > 0 else '' for bar in bar_group])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        st.pyplot(fig)
        
        fig2 , ax =  plt.subplots(figsize=(14, 8))
        sns.countplot(data=df , x='gender', hue='Predict Churn')
        # for container in ax.containers:
        #     ax.bar_label(container)

        st.pyplot(fig2)
      
