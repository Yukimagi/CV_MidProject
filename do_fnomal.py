import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
# 刪除第 80 到 335 欄
df.drop(columns=df.columns[80:336], inplace=True)
##########################################################################
# # 是否正規化
# scaler = MinMaxScaler()
# print(df.iloc[:,:-4])
# df.iloc[:,:-4] = pd.DataFrame(scaler.fit_transform(df.iloc[:,:-4]), columns=df.columns[:-4])
#
# # 將 DataFrame 匯出為 CSV 文件
# df.to_csv('fnormal.txt', index=False, encoding='utf-8')
###########################################################################
# 將 DataFrame 匯出為 CSV 文件
df.to_csv('unformal.txt', index=False, encoding='utf-8')