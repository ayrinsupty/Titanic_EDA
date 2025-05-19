import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page config & dark mode toggle
st.set_page_config(page_title="Titanic EDA & Prediction", layout="wide")

def load_css(dark_mode):
    css_file = "style/dark_mode.css" if dark_mode else "style/light_mode.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Dark mode toggle in sidebar
dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=True)  # Or however you choose the theme
load_css(dark_mode)


# Title and description
st.title("🛳️ Titanic EDA and Survival Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv')
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Dataset")
pclass_filter = st.sidebar.multiselect("Passenger Class", options=sorted(df['Pclass'].unique()), default=sorted(df['Pclass'].unique()))
sex_filter = st.sidebar.multiselect("Sex", options=sorted(df['Sex'].unique()), default=sorted(df['Sex'].unique()))
embarked_filter = st.sidebar.multiselect("Embarked", options=sorted(df['Embarked'].dropna().unique()), default=sorted(df['Embarked'].dropna().unique()))

# Apply filters
filtered_df = df[
    (df['Pclass'].isin(pclass_filter)) &
    (df['Sex'].isin(sex_filter)) &
    (df['Embarked'].isin(embarked_filter))
]

st.subheader("Filtered Dataset")
st.dataframe(filtered_df)
if filtered_df.empty:
    st.warning("⚠️ No data available after applying filters. Please adjust your filter selections.")
    st.stop()

# Correlation Heatmap
st.subheader("Correlation Heatmap")
heatmap_df = filtered_df.select_dtypes(include=['number']).copy()
if 'Title' in heatmap_df.columns:
    heatmap_df.drop('Title', axis=1, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Load pre-trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define numerical columns for scaling
num_cols = ['Age', 'Fare', 'FamilySize']

# Prediction function
def predict_survival(input_data):
    input_df = pd.DataFrame([input_data])
    # Feature engineering
    input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch'] + 1

    # Age bins consistent with training
    input_df['AgeBin'] = pd.cut(
        input_df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
    )

    # Fare bins consistent with training (quantile cut points from training data)
    input_df['FareBin'] = pd.cut(
        input_df['Fare'],
        bins=[-1, 7.91, 14.454, 31, 600],
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )

    # One-hot encoding with drop_first=True
    input_df = pd.get_dummies(input_df, columns=['Sex', 'Embarked', 'AgeBin', 'FareBin'], drop_first=True)

    # Ensure all model features are present in the input
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if model_features is not None:
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]
    else:
        # Fallback: match columns from training dataframe if you saved columns separately
        pass

    # Scale numerical features
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    return pred, proba

# Prediction input form
st.subheader("Survival Prediction Input")
with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], index=2)
    sex = st.selectbox("Sex", options=['male', 'female'])
    age = st.number_input("Age", min_value=0.42, max_value=80.0, value=30.0)
    sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, max_value=8, value=0)
    parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=6, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=513.0, value=32.0)
    embarked = st.selectbox("Embarked", options=['C', 'Q', 'S'], index=2)
    submit = st.form_submit_button("Predict Survival")

    if submit:
        input_data = {
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }
        prediction, probability = predict_survival(input_data)
        st.write(f"**Prediction:** {'Survived' if prediction == 1 else 'Did not survive'}")
        st.write(f"**Survival Probability:** {probability:.2%}")

# Download filtered data as CSV
st.subheader("Download Filtered Dataset")
csv = filtered_df.to_csv(index=False).encode()
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='titanic_filtered_data.csv',
    mime='text/csv'
)

# Final Touch
st.caption("Made with ❤️ by Supty | Titanic EDA and Prediction App")
