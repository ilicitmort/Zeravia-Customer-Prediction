import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Zeravia Customer Predictor",
    page_icon="🛍️",
    layout="centered"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🛍️ Zeravia Customer Purchase Predictor")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to purchase next month."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("zeravia_customer_data.csv")

    # Standardise categorical text
    for col in [
        "Gender",
        "Region",
        "SignupChannel",
        "PreferredCategory"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Standardise inconsistent gender values
    gender_map = {
        "M": "Male",
        "Male": "Male",
        "Female": "Female",
        "F": "Female",
        "Other": "Other"
    }

    # Standardise inconsistent region values
    region_map = {
        "North": "North",
        "N.": "North",
        "South": "South",
        "East": "East",
        "West": "West"
    }

    df["Gender"] = df["Gender"].replace(gender_map)
    df["Region"] = df["Region"].replace(region_map)

    # Remove duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates(
        subset="CustomerID",
        keep="first"
    )

    # Fix invalid ages
    invalid_age = (df["Age"] < 16) | (df["Age"] > 100)
    df.loc[invalid_age, "Age"] = np.nan

    # Fix invalid income
    df.loc[df["AnnualIncome"] < 0, "AnnualIncome"] = np.nan

    # Fill numerical missing values
    numeric_cols = [
        "Age",
        "AnnualIncome",
        "EmailOpenRate",
        "AvgOrderValue",
        "TenureMonths"
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    df["PreferredCategory"] = (
        df["PreferredCategory"]
        .fillna(df["PreferredCategory"].mode()[0])
    )

    return df


df = load_data()


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

# Age group
age_bins = [0, 25, 35, 45, 55, 65, 120]

age_labels = [
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "56-65",
    "66+"
]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    right=True
)


# Engagement level
engagement_score = (
    df["WebsiteVisitsPerMonth"].rank(pct=True) * 0.5
    +
    df["EmailOpenRate"].rank(pct=True) * 0.5
)

df["EngagementLevel"] = pd.cut(
    engagement_score,
    bins=[0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)


# ---------------------------------------------------------
# ONE-HOT ENCODING
# ---------------------------------------------------------

categorical_features = [
    "Gender",
    "Region",
    "SignupChannel",
    "PreferredCategory",
    "AgeGroup",
    "EngagementLevel"
]

modelling_df = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True
)


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

classification_features = [
    "Age",
    "AnnualIncome",
    "TenureMonths",
    "LoyaltyMember",
    "WebsiteVisitsPerMonth",
    "EmailOpenRate",
    "SupportTickets",
    "DiscountUsedPct",
    "NumPurchases",
    "AvgOrderValue"
]

classification_features += [
    c for c in modelling_df.columns
    if c.startswith(
        (
            "Gender_",
            "Region_",
            "SignupChannel_",
            "PreferredCategory_",
            "AgeGroup_",
            "EngagementLevel_"
        )
    )
]


X = modelling_df[classification_features]
y = modelling_df["WillPurchaseNextMonth"]


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

st.header("Customer Information")


age = st.number_input(
    "Age",
    min_value=16,
    max_value=100,
    value=30
)


income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)


tenure = st.number_input(
    "Tenure (Months)",
    min_value=1,
    max_value=72,
    value=24
)


loyalty = st.selectbox(
    "Loyalty Member",
    ["No", "Yes"]
)

loyalty_value = 1 if loyalty == "Yes" else 0


website_visits = st.number_input(
    "Website Visits Per Month",
    min_value=0.0,
    value=6.0
)


email_open_rate = st.number_input(
    "Email Open Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.35
)


support_tickets = st.number_input(
    "Support Tickets",
    min_value=0.0,
    value=0.0
)


discount_used = st.number_input(
    "Discount Used %",
    min_value=0.0,
    max_value=1.0,
    value=0.20
)


num_purchases = st.number_input(
    "Number of Purchases",
    min_value=0.0,
    value=4.0
)


avg_order_value = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=80.0
)


gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)


region = st.selectbox(
    "Region",
    ["East", "North", "South", "West"]
)


signup_channel = st.selectbox(
    "Signup Channel",
    [
        "Organic",
        "Referral",
        "Search",
        "Social Media",
        "Other"
    ]
)


preferred_category = st.selectbox(
    "Preferred Category",
    [
        "Electronics",
        "Fashion",
        "Groceries",
        "Beauty",
        "Sports",
        "Home & Living"
    ]
)


# ---------------------------------------------------------
# CALCULATE DERIVED FEATURES
# ---------------------------------------------------------

# Determine age group
if age <= 25:
    age_group = "18-25"
elif age <= 35:
    age_group = "26-35"
elif age <= 45:
    age_group = "36-45"
elif age <= 55:
    age_group = "46-55"
elif age <= 65:
    age_group = "56-65"
else:
    age_group = "66+"


# Approximate engagement level
# We compare the customer's engagement with
# the training-data distributions.

visit_percentile = (
    df["WebsiteVisitsPerMonth"] <= website_visits
).mean()

email_percentile = (
    df["EmailOpenRate"] <= email_open_rate
).mean()

engagement_score_user = (
    visit_percentile * 0.5
    +
    email_percentile * 0.5
)

if engagement_score_user <= 0.33:
    engagement_level = "Low"
elif engagement_score_user <= 0.66:
    engagement_level = "Medium"
else:
    engagement_level = "High"


# ---------------------------------------------------------
# CREATE USER DATAFRAME
# ---------------------------------------------------------

user_data = pd.DataFrame({
    "Age": [age],
    "AnnualIncome": [income],
    "TenureMonths": [tenure],
    "LoyaltyMember": [loyalty_value],
    "WebsiteVisitsPerMonth": [website_visits],
    "EmailOpenRate": [email_open_rate],
    "SupportTickets": [support_tickets],
    "DiscountUsedPct": [discount_used],
    "NumPurchases": [num_purchases],
    "AvgOrderValue": [avg_order_value],
    "Gender": [gender],
    "Region": [region],
    "SignupChannel": [signup_channel],
    "PreferredCategory": [preferred_category],
    "AgeGroup": [age_group],
    "EngagementLevel": [engagement_level]
})


# One-hot encode user data
user_data = pd.get_dummies(
    user_data,
    columns=categorical_features,
    drop_first=True
)


# Make sure user data has exactly the same
# columns as the training data.

user_data = user_data.reindex(
    columns=classification_features,
    fill_value=0
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("🔮 Predict Purchase Probability"):

    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]

    probability = model.predict_proba(
        user_scaled
    )[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "🟢 This customer is likely to purchase next month."
        )

    else:

        st.warning(
            "🔴 This customer is unlikely to purchase next month."
        )

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))
