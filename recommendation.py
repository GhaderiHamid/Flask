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

# 2️⃣ دریافت اطلاعات خرید کاربران + دسته‌بندی محصولات
query = """
SELECT od.product_id, o.user_id, p.category_id
FROM orders o
JOIN order_details od ON o.id = od.order_id
JOIN products p ON od.product_id = p.id
"""

cursor.execute(query)
data = pd.DataFrame(cursor.fetchall(), columns=['product_id', 'user_id', 'category_id'])

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

# 7️⃣ تابع پیشنهاد محصول با تنوع دسته‌بندی
def recommend_products(user_id, num_neighbors=10, max_recommendations=30, max_per_category=2):
    if user_id not in pivot_table.index:
        return []

    # پیدا کردن کاربران مشابه
    distances, indices = model.kneighbors([pivot_table.loc[user_id]], n_neighbors=min(num_neighbors, num_samples))

    # محصولات خریداری‌شده توسط کاربر هدف
    user_products = set(data[data['user_id'] == user_id]['product_id'])

    # جمع‌آوری محصولات کاربران مشابه
    similar_users = pivot_table.index[indices[0]]
    all_new_products = pd.DataFrame()

    for neighbor_id in similar_users:
        neighbor_data = data[data['user_id'] == neighbor_id]
        new_data = neighbor_data[~neighbor_data['product_id'].isin(user_products)]
        all_new_products = pd.concat([all_new_products, new_data])

    # حذف تکراری‌ها
    all_new_products.drop_duplicates(subset='product_id', inplace=True)

    # گروه‌بندی بر اساس دسته‌بندی و انتخاب محدود از هر دسته
    recommended_products = []
    grouped = all_new_products.groupby('category_id')

    for _, group in grouped:
        selected = group.head(max_per_category)
        recommended_products.extend(selected['product_id'].tolist())

        if len(recommended_products) >= max_recommendations:
            break

    return list(map(int, recommended_products[:max_recommendations]))

# 8️⃣ راه‌اندازی API با Flask برای ارتباط با لاراول
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = int(request.args.get("user_id"))
        limit = int(request.args.get("limit", 30))  # تعداد کل پیشنهادها
        per_category = int(request.args.get("per_category", 2))  # تعداد از هر دسته‌بندی
    except (TypeError, ValueError):
        return jsonify({"error": "پارامترهای ورودی نامعتبر هستند!"}), 400

    recommendations = recommend_products(
        user_id,
        num_neighbors=10,
        max_recommendations=limit,
        max_per_category=per_category
    )

    return jsonify({"user_id": user_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
