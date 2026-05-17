"""
Data cleaning module for DoorDash fraud detection.
Handles missing value analysis, imputation, and categorical encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Columns with high null rates that correlate with fraud
HIGH_NULL_COLS = [
    'email_address_age_days', 'email_domain_label', 'email_domain_tld_label',
    'email_username_length', 'latest_order_amount_usd', 'latest_item_category',
    'latest_item_quantity', 'latest_changed_password',
    'latest_delivery_address_name_length', 'latest_delivery_address_region_label',
    'latest_delivery_address_fraction_vowels',
    'per_day_add_item_to_cart', 'per_day_transactions', 'per_day_purchase_total',
    'per_day_unique_billing_last4', 'per_week_purchase_total', 'per_week_update_account',
    'max_item_count', 'per_month_logout', 'per_month_page_activity',
    'per_month_transactions', 'num_unique_delivery_addresses', 'days_since_first_transaction'
]

# Activity/count columns where null means "never happened"
ZERO_FILL_COLS = [
    'per_day_add_item_to_cart', 'per_day_transactions', 'per_day_purchase_total',
    'per_day_unique_billing_last4', 'per_day_payment_method_change',
    'per_day_devices_per_user', 'per_week_purchase_total', 'per_week_unique_ips',
    'per_week_update_account', 'per_week_payment_method_change',
    'max_item_count', 'per_month_logout', 'per_month_page_activity',
    'per_month_transactions', 'num_unique_delivery_addresses',
    'days_since_first_transaction'
]

# Numeric profile columns to fill with median
MEDIAN_FILL_COLS = [
    'email_address_age_days', 'email_domain_label', 'email_domain_tld_label',
    'email_username_length', 'latest_order_amount_usd', 'latest_item_quantity',
    'latest_item_tag_count', 'latest_delivery_address_name_length',
    'latest_delivery_address_region_label', 'latest_delivery_address_fraction_vowels'
]

# Categorical columns
CATEGORICAL_COLS = ['latest_item_category', 'latest_item_product_title', 'latest_changed_password']


def create_missing_indicators(df):
    """
    Create binary _is_missing flags for columns with high null rates.
    Missingness correlates with fraud, so these are useful features.
    """
    for col in HIGH_NULL_COLS:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    return df


def impute_zeros(df):
    """
    Fill activity/count columns with 0.
    Null in these columns means "this never happened."
    """
    cols = [c for c in ZERO_FILL_COLS if c in df.columns]
    df[cols] = df[cols].fillna(0)
    return df


def impute_medians(df):
    """
    Fill numeric profile columns with their median values.
    """
    cols = [c for c in MEDIAN_FILL_COLS if c in df.columns]
    df[cols] = df[cols].fillna(df[cols].median())
    return df


def impute_categoricals(df):
    """
    Fill categorical columns with explicit labels.
    """
    if 'latest_item_category' in df.columns:
        df['latest_item_category'] = df['latest_item_category'].fillna('unknown')
    if 'latest_item_product_title' in df.columns:
        df['latest_item_product_title'] = df['latest_item_product_title'].fillna('unknown')
    if 'latest_changed_password' in df.columns:
        df['latest_changed_password'] = df['latest_changed_password'].fillna('never')
    return df


def encode_categoricals(df):
    """
    Label encode categorical columns.
    Converts to string first to handle mixed types (bool + str).
    """
    le_dict = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    return df, le_dict


def clean_data(df):
    """
    Full data cleaning pipeline.

    Steps:
    1. Create missing indicator flags (before imputation)
    2. Impute activity columns with 0
    3. Impute numeric profile columns with median
    4. Impute categorical columns with explicit labels
    5. Encode categoricals

    Returns:
        tuple: (cleaned DataFrame, label encoder dict)
    """
    df = create_missing_indicators(df)
    df = impute_zeros(df)
    df = impute_medians(df)
    df = impute_categoricals(df)
    df, le_dict = encode_categoricals(df)

    # Verify no nulls remain
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"Warning: {remaining_nulls} null values remain after cleaning")

    return df, le_dict
