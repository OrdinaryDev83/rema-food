from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

customers_df = pd.read_csv('../data/train_customers.csv')
vendors_df = pd.read_csv('../data/vendors.csv')
orders_df = pd.read_csv('../data/orders.csv')

to_remove = [
            "vendor_category_en",
            "authentication_id",
            "OpeningTime",
            "OpeningTime2",
            "open_close_flags",
            "created_at",
            "updated_at",
            "commission",
            "saturday_to_time1",
            "saturday_from_time2",
            "saturday_from_time1",
            "saturday_to_time2",
            "thursday_to_time1",
            "thursday_from_time1",
            "thursday_from_time2",
            "thursday_to_time2",
            "tuesday_to_time1",
            "tuesday_from_time2",
            "tuesday_from_time1",
            "tuesday_to_time2",
            "monday_to_time1",
            "monday_from_time1",
            "monday_from_time2",
            "monday_to_time2",
            "sunday_to_time1",
            "sunday_from_time1",
            "sunday_from_time2",
            "sunday_to_time2",
            "friday_to_time1",
            "friday_from_time1",
            "friday_from_time2",
            "friday_to_time2",
            "wednesday_to_time1",
            "wednesday_from_time1",
            "wednesday_from_time2",
            "wednesday_to_time2",
            "one_click_vendor",
            "country_id",
            "city_id",
            "display_orders",
            "device_type",
            "is_akeed_delivering",
            "language",
            "rank"
            ]

df_vendors = df_vendors.drop(to_remove, axis=1)
# remove all unverified accounts
df_vendors = df_vendors[df_vendors["verified"] == 1]
df_vendors = df_vendors.drop("verified", axis=1)
df_vendors["primary_tags"] = df_vendors["primary_tags"].fillna("{\"primary_tags\":\"0\"}").astype(str).map(lambda x: x.split("\"")[3]).astype(int)
df_vendors["primary_tags"].value_counts()

df_customers = df_customers.drop(["language", "dob"], axis=1)

# gender string to integer, 0 for male, 1 for female, 2 for unknown
df_customers["gender"] = df_customers["gender"].str.strip()
df_customers["gender"] = df_customers["gender"].str.upper()
df_customers["gender"] = df_customers["gender"].map({'MALE': 0, 'FEMALE': 1})
df_customers["gender"] = df_customers.gender.fillna(2).astype(int)

df_customers = df_customers[(df_customers.verified == 1) & (df_customers.status == 1)].drop(["verified", "status", "created_at", "updated_at"], axis=1)

# Assume we have the following dataframes: customers_df, vendors_df, orders_df

# Preprocessing and creating mappings
customer_ids = LabelEncoder().fit_transform(customers_df['akeed_customer_id'])
vendor_ids = LabelEncoder().fit_transform(vendors_df['id'])

# Creating dictionaries for quick access
customer_mapping = dict(zip(customers_df['akeed_customer_id'], customer_ids))
vendor_mapping = dict(zip(vendors_df['id'], vendor_ids))

# Prepare orders data
orders_df['customer_id'] = orders_df['customer_id'].map(customer_mapping)
orders_df['vendor_id'] = orders_df['vendor_id'].map(vendor_mapping)

# Assume that we have a function to encode 'gender' and 'vendor_tag_name' to numerical vectors
# For simplicity, let's just map them to integers here
customers_df['gender'] = LabelEncoder().fit_transform(customers_df['gender'].fillna(''))

# Creating model
embedding_size = 50

# Inputs
customer_input = Input(shape=(1,))
vendor_input = Input(shape=(1,))

# Embeddings for collaborative filtering
customer_embedding = Embedding(len(customer_ids), embedding_size, input_length=1)(customer_input)
vendor_embedding = Embedding(len(vendor_ids), embedding_size, input_length=1)(vendor_input)

# Flatten embeddings
customer_vec = Flatten()(customer_embedding)
vendor_vec = Flatten()(vendor_embedding)

# Additional inputs for the features
gender_input = Input(shape=(1,), dtype='int32')
vendor_tag_name_input = Input(shape=(1,), dtype='int32')

# One-hot encoding for categorical features
gender_embedding = Embedding(input_dim=2, output_dim=embedding_size)(gender_input)  # Assuming 'gender' is binary
vendor_tag_name_embedding = Embedding(input_dim=100, output_dim=embedding_size)(vendor_tag_name_input)  # Replace 'num_vendor_tags' with the actual number of vendor tags

# Flatten the embeddings
gender_vec = Flatten()(gender_embedding)
vendor_tag_name_vec = Flatten()(vendor_tag_name_embedding)

# Concatenate all vectors
input_vec = Concatenate()([customer_vec, vendor_vec, gender_vec, vendor_tag_name_vec])

# Dense layers
dense = Dense(128, activation='relu')(input_vec)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='sigmoid')(dense)

# Compile model
model = Model(inputs=[customer_input, vendor_input, gender_input, vendor_tag_name_input], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=Adam())


def generate_training_data(orders_df, customers_df, vendors_df, customer_ids, vendor_ids):
    # Create a set of all (customer_id, vendor_id) tuples that have an order
    order_tuples = set(zip(orders_df['customer_id'], orders_df['vendor_id']))

    # Prepare data for all possible combinations of customer-vendor pairs
    all_customer_ids = np.repeat(customer_ids, len(vendor_ids))
    all_vendor_ids = np.tile(vendor_ids, len(customer_ids))

    # Match the gender and vendor_tag_name to each customer and vendor
    all_genders = np.repeat(customers_df['gender'], len(vendor_ids))
    all_vendor_tags = np.repeat(vendors_df['vendor_tag_name'], len(customer_ids))

    # Create a binary label indicating whether an order exists for each pair
    labels = np.array([(customer_id, vendor_id) in order_tuples 
                       for customer_id, vendor_id in zip(all_customer_ids, all_vendor_ids)])

    # Split the data into training and validation sets
    customer_train, customer_val, vendor_train, vendor_val, gender_train, gender_val, vendor_tag_train, vendor_tag_val, y_train, y_val = train_test_split(
        all_customer_ids, all_vendor_ids, all_genders, all_vendor_tags, labels, test_size=0.1, random_state=42)

    return (customer_train, vendor_train, gender_train, vendor_tag_train), y_train, (customer_val, vendor_val, gender_val, vendor_tag_val), y_val

X_train, y_train, X_val, y_val = generate_training_data(orders_df, customers_df, vendors_df, customer_ids, vendor_ids)

model.fit([X_train[0], X_train[1], X_train[2], X_train[3]], y_train, epochs=10, validation_split=0.1)