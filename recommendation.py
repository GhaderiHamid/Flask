import mysql.connector
import os
import pandas as pd
import random
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

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
SELECT o.user_id, od.product_id, COUNT(*) AS rating, p.category_id
FROM orders o
JOIN order_details od ON o.id = od.order_id
JOIN products p ON od.product_id = p.id
GROUP BY o.user_id, od.product_id, p.category_id
"""

cursor.execute(query)
data = pd.DataFrame(cursor.fetchall(), columns=['user_id', 'product_id', 'rating', 'category_id'])

cursor.close()
db.close()

# 3️⃣ بررسی اینکه دیتابیس خالی نباشد
if data.empty:
    print("❌ خطا: دیتابیس خرید کاربران خالی است!")
    exit()

# 4️⃣ ساخت دیتافریم برای SVD
df_svd = data[['user_id', 'product_id', 'rating']]

reader = Reader(rating_scale=(1, df_svd['rating'].max()))
dataset = Dataset.load_from_df(df_svd, reader)
trainset, testset = train_test_split(dataset, test_size=0.2)

svd_model = SVD()
svd_model.fit(trainset)

# 5️⃣ ساخت pivot table برای NearestNeighbors
pivot_table = data.pivot(index='user_id', columns='product_id', values='product_id').fillna(0)
num_samples = pivot_table.shape[0]

model_nn = NearestNeighbors(n_neighbors=min(10, num_samples), metric='cosine', algorithm='brute')
model_nn.fit(pivot_table)

# 6️⃣ تابع پیشنهاد با NearestNeighbors
def recommend_with_neighbors(user_id, num_neighbors=10, max_recommendations=30, max_per_category=2):
    if user_id not in pivot_table.index:
        return []

    distances, indices = model_nn.kneighbors([pivot_table.loc[user_id]], n_neighbors=min(num_neighbors, num_samples))
    user_products = set(data[data['user_id'] == user_id]['product_id'])

    similar_users = pivot_table.index[indices[0]]
    all_new_products = pd.DataFrame()

    for neighbor_id in similar_users:
        neighbor_data = data[data['user_id'] == neighbor_id]
        new_data = neighbor_data[~neighbor_data['product_id'].isin(user_products)]
        all_new_products = pd.concat([all_new_products, new_data])

    all_new_products.drop_duplicates(subset='product_id', inplace=True)

    recommended_products = []
    grouped = all_new_products.groupby('category_id')
    category_ids = list(grouped.groups.keys())
    random.shuffle(category_ids)

    for category_id in category_ids:
        group = grouped.get_group(category_id)
        if group.empty:
            continue
        shuffled_group = group.sample(frac=1)
        selected = shuffled_group.head(max_per_category)
        recommended_products.extend(selected['product_id'].tolist())

        if len(recommended_products) >= max_recommendations:
            break

    return list(map(int, recommended_products[:max_recommendations]))

# 7️⃣ تابع پیشنهاد با SVD
def recommend_with_svd(user_id, top_n=30):
    user_products = set(df_svd[df_svd['user_id'] == user_id]['product_id'])
    all_products = set(df_svd['product_id'].unique())
    candidate_products = list(all_products - user_products)

    scored = []
    for pid in candidate_products:
        pred = svd_model.predict(user_id, pid)
        scored.append((pid, pred.est))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [int(pid) for pid, _ in scored[:top_n]]

# 8️⃣ پیشنهاد عمومی برای fallback
def recommend_popular(limit=30):
    popular = df_svd.groupby('product_id')['rating'].sum().sort_values(ascending=False)
    return list(map(int, popular.head(limit).index))

# 9️⃣ تابع ترکیبی هوشمند
def hybrid_recommend(user_id, limit=30, per_category=2):
    user_data = df_svd[df_svd['user_id'] == user_id]
    if len(user_data) >= 5:
        return recommend_with_svd(user_id, top_n=limit)
    elif user_id in pivot_table.index:
        return recommend_with_neighbors(user_id, max_recommendations=limit, max_per_category=per_category)
    else:
        return recommend_popular(limit)

# 🔟 راه‌اندازی API با Flask
app = Flask(__name__)

@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend(user_id):
    try:
        limit = int(request.args.get("limit", 30))
        per_category = int(request.args.get("per_category", 2))
    except (TypeError, ValueError):
        return jsonify({"error": "پارامترهای ورودی نامعتبر هستند!"}), 400

    recommendations = hybrid_recommend(user_id, limit=limit, per_category=per_category)

    return jsonify({"user_id": user_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
