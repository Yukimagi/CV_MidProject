from flask import Flask, render_template, jsonify, request
import random
import pandas as pd
import os
import ast

# 設定目前工作目錄為這個程式碼所在的目錄
current_file_path = os.path.abspath(__file__)

print(f"Current file path: {current_file_path}")

app = Flask(__name__)

# 50 個類別名稱的清單
categories = [
    "AncestorDinoArt", "Archit", "Baseball", "Basketball", "Beach", "Billiardsball", 
    "Bus", "BWimage", "Car", "Cartoon", "Castle", "Citynight", "ClassicalPainting", 
    "Cropcycle", "DeerAntelope", "Desert", "Dog", "Doors", "Eagle", "Elephant", "F1", 
    "Feasts", "Flower", "Grass", "Group", "Indoor", "Lion", "Masks", "Model", "Mountain", 
    "Owls", "Penguin", "Plane", "Planet", "Pumpkin", "RacingMotor", "Satelliteimage", 
    "Sculpt", "Ship", "Sky", "Soccer", "Stalactite", "SubSea", "Sunflower", "Sunset", 
    "Surfs", "Tennis", "Tiger", "Volleyball", "Waterfall"
]


@app.route("/index.html")
def index():
    return render_template("index.html", categories=categories)

@app.route('/')
def root():
    return render_template('index.html', categories=categories)

@app.route('/random_image/<category>')
def random_image(category):
    # 確保類別存在
    if category not in categories:
        return jsonify({"error": "Invalid category"}), 404

    # 隨機生成1到200之間的數字
    random_number = random.randint(1, 200)
    formatted_number = f"{random_number:03}"
    image_path = f"/static/{category}/{category}_{formatted_number}.jpg"
    
    filename = f"{category}_{formatted_number}.jpg"  # 新增檔名

    return jsonify({"image_path": image_path, "filename": filename})  # 回傳檔名

@app.route('/predict_image', strict_slashes=False)
def predict_image():
    category = request.args.get('category')
    filename = request.args.get('filename')
    normalize = request.args.get('normalize')
    metric = request.args.get('metric')
    
    # 檢查所有必要參數
    if not category or not filename or not normalize or not metric:
        return jsonify({"error": "Missing parameters"}), 400

    # 建立 CSV 文件名稱並嘗試讀取
    csv_file = os.path.join(app.root_path, 'static', f"{normalize}_{metric}_accuracy_output.csv")
    
    print(f"現在的位置:{csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

    # 嘗試找到該圖像名稱並返回前十張相似圖像
    row = df[df['Output'].apply(lambda x: filename == ast.literal_eval(x)[0])]
    print(row)
    
    if row.empty:
        return jsonify({"error": "Image not found in CSV"}), 404

    similar_images = ast.literal_eval(row['Output'].values[0])[1:]  # 第一欄為隨機圖片，後十項為相似圖片
    accuracy = row['Accuracy'].values[0]
    
    for img in similar_images:
        print("cate:\n")
        print(img.split('_')[0])
        print("圖:\n")
        print(img)
# Pass the data to the template
    return render_template("similar_images.html", 
                           category=category,
                           filename=filename,
                           normalize=normalize,
                           metric=metric,
                           categories=categories,
                           random_image_path=f"/static/{category}/{filename}",
                           similar_images=[{"filename": img, "path": f"/static/{img.split('_')[0]}/{img}", "accuracy": accuracy} for img in similar_images])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1121, debug=True)
