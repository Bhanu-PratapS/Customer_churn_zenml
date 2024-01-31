import streamlit as st
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
from src.logger import logger

def main():
    st.title("Machine Learning Pipeline project")

    # Use provided unique values for each column
    age_options = [90, 82, 66, 54, 41, 34, 38, 74, 68, 45, 52, 32, 51, 46, 57, 22, 37, 29, 61, 21, 33, 49, 23, 59, 60, 63, 53, 44, 43, 71, 48, 73, 67, 40, 50, 42, 39, 55, 47, 31, 58, 62, 36, 72, 78, 83, 26, 70, 27, 35, 81, 65, 25, 28, 56, 69, 20, 30, 24, 64, 75, 19, 77, 80, 18, 17, 76, 79, 88, 84, 85, 86, 87]
    age = st.selectbox("Select Age:", age_options)

    workclass_options = ['Private', 'State-gov', 'Federal-gov', 'Self-emp-not-inc', 'Self-emp-inc', 'Local-gov', 'Without-pay', 'Never-worked']
    workclass = st.selectbox("Select Workclass:", workclass_options)

    education_options = ['HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate', 'Prof-school', 'Bachelors', 'Masters', '11th', 'Assoc-acdm', 'Assoc-voc', '1st-4th', '5th-6th', '12th', '9th', 'Preschool']
    education = st.selectbox("Select Education:", education_options)

    education_num_options = [9, 10, 4, 6, 16, 15, 13, 14, 7, 12, 11, 2, 3, 8, 5, 1]
    education_num = st.selectbox("Select Education Number:", education_num_options)

    marital_status_options = ['Widowed', 'Divorced', 'Separated', 'Never-married', 'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
    marital_status = st.selectbox("Select Marital Status:", marital_status_options)

    occupation_options = ['Exec-managerial', 'Machine-op-inspct', 'Prof-specialty', 'Other-service', 'Adm-clerical', 'Craft-repair', 'Transport-moving', 'Handlers-cleaners', 'Sales', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
    occupation = st.selectbox("Select Occupation:", occupation_options)

    relationship_options = ['Not-in-family', 'Unmarried', 'Own-child', 'Other-relative', 'Husband', 'Wife']
    relationship = st.selectbox("Select Relationship:", relationship_options)

    race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo']
    race = st.selectbox("Select Race:", race_options)

    sex_options = ['Female', 'Male']
    sex = st.selectbox("Select Sex:", sex_options)

    hours_per_week_options = [40, 18, 45, 20, 60, 35, 55, 76, 50, 42, 25, 32, 90, 48, 15, 70, 52, 72, 39, 6, 65, 12, 80, 67, 99, 30, 75, 26, 36, 10, 84, 38, 62, 44, 8, 28, 59, 5, 24, 57, 34, 37, 46, 56, 41, 98, 43, 63, 1, 47, 68, 54, 2, 16, 9, 3, 4, 33, 23, 22, 64, 51, 19, 58, 53, 96, 66, 21, 7, 13, 27, 11, 14, 77, 31, 78, 49, 17, 85, 87, 88, 73, 89, 97, 94, 29, 82, 86, 91, 81, 92, 61, 74, 95]
    hours_per_week = st.selectbox("Select Hours per Week:", hours_per_week_options)

    income_options = ['<=50K', '>50K']
    income = st.selectbox("Select Income:", income_options)

    if st.button("Predict"):
        data = CustomClass(
            age=age,
            education_num=education_num,
            hours_per_week=hours_per_week,
            workclass=workclass,
            education=education,
            marital_status=marital_status,
            occupation=occupation,
            relationship=relationship,
            race=race,
            sex=sex,
            income=income
        )

        final_data = data.get_data_dataframe()
        pipeline_prediction = PredictionPipeline()
        pred = pipeline_prediction.predict(final_data)

        if pred == 0:
            st.success("The person is predicted to earn less than 50K.")
        elif pred == 1:
            st.success("The person is predicted to earn more than 50K.")

if __name__ == "__main__":
    main()
