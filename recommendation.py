import os
import mysql.connector
import pandas as pd
from flask import Flask, request, jsonify
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import coo_matrix

# 1️⃣ اتصال به دیتابیس و دریافت داده‌ها
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

cursor = db.cursor()
cursor.execute("""
SELECT o.user_id, od.product_id
FROM orders o
JOIN order_details od ON o.id = od.order_id
""")
rows = cursor.fetchall()
cursor.close()
db.close()

df = pd.DataFrame(rows, columns=["user_id", "product_id"])
if df.empty:
    raise Exception("❌ دیتابیس خالی است!")

# 2️⃣ ساخت Dataset برای LightFM
users = df["user_id"].unique()
items = df["product_id"].unique()
interactions = list(df.itertuples(index=False, name=None))

dataset = Dataset()
dataset.fit(users, items)
dataset.fit_partial(users=users, items=items)

(interactions_matrix, _) = dataset.build_interactions(interactions)

# 3️⃣ تقسیم داده‌ها به ۸۰٪ آموزش و ۲۰٪ تست
train, test = random_train_test_split(interactions_matrix, test_percentage=0.2)

# 4️⃣ آموزش مدل LightFM
model = LightFM(loss="warp")
model.fit(train, epochs=10, num_threads=2)

# 5️⃣ ساخت نگاشت‌های داخلی
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

# 6️⃣ تابع پیشنهاد محصول برای کاربر خاص
def recommend_for_user(user_id, top_n=30):
    if user_id not in user_id_map:
        return []

    internal_uid = user_id_map[user_id]
    all_item_ids = list(item_id_map.keys())
    internal_iids = [item_id_map[iid] for iid in all_item_ids]

    scores = model.predict(internal_uid, internal_iids)
    scored_items = sorted(zip(all_item_ids, scores), key=lambda x: x[1], reverse=True)

    # حذف محصولاتی که کاربر قبلاً خریده
    user_purchases = df[df["user_id"] == user_id]["product_id"].tolist()
    recommendations = [int(pid) for pid, _ in scored_items if pid not in user_purchases]

    return recommendations[:top_n]

# 7️⃣ راه‌اندازی Flask API
app = Flask(__name__)

@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    try:
        limit = int(request.args.get("limit", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "پارامتر نامعتبر است!"}), 400

    recommendations = recommend_for_user(user_id, top_n=limit)
    return jsonify({"user_id": user_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
