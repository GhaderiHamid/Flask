import mysql.connector
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors

# 1️⃣ اتصال به پایگاه داده MySQL
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

cursor = db.cursor()

# 2️⃣ دریافت اطلاعات خرید کاربران از دیتابیس
query = """
SELECT od.product_id, o.user_id
FROM orders o
JOIN order_details od ON o.id = od.order_id
"""

cursor.execute(query)
data = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'user_id'])

cursor.close()
db.close()

# 3️⃣ بررسی اینکه دیتابیس خالی نباشد
if data.empty:
    print("❌ خطا: دیتابیس خرید کاربران خالی است!")
    exit()

# 4️⃣ پردازش داده‌ها برای مدل پیشنهادگر
data.drop_duplicates(inplace=True)
pivot_table = data.pivot(index='user_id', columns='product_id', values='product_id').fillna(0)

# 5️⃣ بررسی حداقل تعداد کاربران برای مدل پیشنهادگر
num_samples = pivot_table.shape[0]
if num_samples < 2:
    print("❌ خطا: تعداد کاربران برای آموزش مدل کافی نیست!")
    exit()

# 6️⃣ ایجاد مدل پیشنهادگر با تعداد همسایگان بیشتر
model = NearestNeighbors(n_neighbors=min(10, num_samples), metric='cosine', algorithm='brute')
model.fit(pivot_table)

# 7️⃣ تابع پیشنهاد محصول فقط برای محصولات جدید
def recommend_products(user_id, num_neighbors=10, max_recommendations=10):
    if user_id not in pivot_table.index:
        return []

    # پیدا کردن کاربران مشابه
    distances, indices = model.kneighbors([pivot_table.loc[user_id]], n_neighbors=min(num_neighbors, num_samples))

    # محصولات خریداری‌شده توسط کاربر هدف
    user_products = set(data[data['user_id'] == user_id]['product_id'])

    # جمع‌آوری محصولات کاربران مشابه
    similar_users = pivot_table.index[indices[0]]
    recommended_products = set()

    for neighbor_id in similar_users:
        neighbor_products = set(data[data['user_id'] == neighbor_id]['product_id'])
        new_products = neighbor_products - user_products
        recommended_products.update(new_products)

    # محدود کردن تعداد پیشنهادها
    return list(map(int, list(recommended_products)[:max_recommendations]))

# 8️⃣ راه‌اندازی API با Flask برای ارتباط با لاراول
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = int(request.args.get("user_id"))
        limit = int(request.args.get("limit", 10))  # تعداد پیشنهادها به صورت اختیاری
    except (TypeError, ValueError):
        return jsonify({"error": "پارامترهای ورودی نامعتبر هستند!"}), 400

    recommendations = recommend_products(user_id, num_neighbors=10, max_recommendations=limit)
    return jsonify({"user_id": user_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
