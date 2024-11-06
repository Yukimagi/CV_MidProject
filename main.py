import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine, euclidean
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 資料夾路徑
folder_path = 'raw features'
data = []
# 瀏覽資料夾內的每一個檔案
for file_name in os.listdir(folder_path):
    # 檢查檔案是否為txt檔
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)

        # 打開並讀取每一行，略過第一行
        with open(file_path, 'r', encoding='utf-8') as file:
            # 使用 enumerate() 並從第二行開始讀取
            for idx, line in enumerate(file):
                if idx == 0:
                    continue  # 跳過第一行
                # 將每行以空格分隔成清單並加入data列表
                row_data = line.strip().split(" ")
                data.append(np.array(row_data))
# 將數據轉換為DataFrame
df = pd.DataFrame(data)
# 刪除第 80 到 355 欄
df.drop(columns=df.columns[80:336], inplace=True)
contrast = df.drop(columns=df.columns[-4:]).astype(int)
contrast.fillna(0, inplace=True)
##########################################################################
# # 是否正規化
# scaler = MinMaxScaler()
# contrast = pd.DataFrame(scaler.fit_transform(contrast), columns=contrast.columns)
# # 將 DataFrame 匯出為 CSV 文件
# df.to_csv('fnormal.csv', index=False, encoding='utf-8')
###########################################################################
# 將 DataFrame 匯出為 CSV 文件
#df.to_csv('unformal.csv', index=False, encoding='utf-8')
ans = df.drop(columns=df.columns[:-1])

# 創建一個空的列表來儲存準確率結果
accuracies = []

# 逐行計算
for i in tqdm(range(len(df)), desc="計算相似度", unit="row"):
    # 取得當前行的資料和答案
    current_row_data = contrast.iloc[i]  # 前 201 欄
    current_answer = ans.iloc[i, 0]  # 第 202 欄（答案）
    ###########################################################################
    # 更改euclidean，cosine
    # 計算 Pearson 相關係數，並找出相似度
    Pearson_similarities = contrast.corrwith(current_row_data, numeric_only=True, axis=1, method='pearson')
    # 計算 cosine 相關係數，並找出相似度
    #cosine_similarities = contrast.apply(lambda row: 1 - cosine(row, current_row_data), axis=1)
    # 計算 euclidean 相關係數，並找出相似度
    #euclidean_similarities = contrast.apply(lambda row: 1 - euclidean(row, current_row_data), axis=1)
    ###########################################################################
    # 找出前 10 相似的行（排除自己）
    top_10_indices = Pearson_similarities.nlargest(11).index[1:]  # 取前 11 名，排除自己

    # 檢查這些行的答案是否與當前行相同
    matches = (df.iloc[top_10_indices, -1] == current_answer).sum()  # 第 202 欄（答案）

    # 計算準確率
    accuracy = matches / 10  # 符合的比例
    accuracies.append(accuracy)

# 計算總準確率
overall_accuracy = sum(accuracies) / len(accuracies)
print(f'總準確率: {overall_accuracy:.2%}')
