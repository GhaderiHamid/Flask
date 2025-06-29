import mysql.connector
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors

# 1️⃣ اتصال به پایگاه داده MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",  # نام کاربری خود را تنظیم کنید
    password="",  # رمزعبور پایگاه داده را تنظیم کنید
    database="store"  # نام دیتابیس شما
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

# 6️⃣ ایجاد مدل پیشنهادگر
model = NearestNeighbors(n_neighbors=min(5, num_samples), metric='cosine', algorithm='brute')
model.fit(pivot_table)

def recommend_products(user_id):
    if user_id not in pivot_table.index:
        return []  # اگر کاربر در دیتابیس نبود، مقدار خالی برگردان

    distances, indices = model.kneighbors([pivot_table.loc[user_id]])
    
    # تبدیل مقادیر از int64 به int برای جلوگیری از خطای JSON
    recommended_product_ids = [int(pivot_table.columns[i]) for i in indices[0]]
    
    return recommended_product_ids

# 7️⃣ راه‌اندازی API با Flask برای ارتباط با لاراول
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = int(request.args.get("user_id"))
    recommendations = recommend_products(user_id)

    # تبدیل مقادیر به لیست عدد صحیح برای JSON
    return jsonify({"user_id": user_id, "recommendations": list(map(int, recommendations))})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)