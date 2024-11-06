import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import csv

from scipy.spatial.distance import cosine, euclidean
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

df = pd.read_csv('fnormal.txt')
#df = pd.read_csv('unformal.txt')

# 創建一個空的列表來儲存準確率結果
accuracies = []
outputs = []
contrast = df.drop(columns=df.columns[-4:]).astype(int)
ans = df.drop(columns=df.columns[:-3])

print(contrast.shape[1])  # 顯示欄位數量

# 打開一個 CSV 檔案，進行寫入操作
with open('formal_eu_accuracy_output.csv', 'w', newline='') as csvfile:
    fieldnames = ['Output', 'Accuracy']  # 定義欄位名稱
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()  # 寫入欄位名稱

# 逐行計算
    for i in tqdm(range(len(df)), desc="計算相似度", unit="row"):
        # 取得當前行的資料和答案
        output = []
        output.append(ans.iloc[i, 0])
        current_row_data = contrast.iloc[i]  # 前 221 欄
        current_answer = ans.iloc[i, 2]  # 第 222 欄（答案）
        ###########################################################################
        # 更改euclidean，cosine
        # 計算 Pearson 相關係數，並找出相似度
        #Pearson_similarities = contrast.corrwith(current_row_data, numeric_only=True, axis=1, method='pearson')
        # 計算 cosine 相關係數，並找出相似度
        #cosine_similarities = contrast.apply(lambda row: 1 - cosine(row, current_row_data), axis=1)
        # 計算 euclidean 相關係數，並找出相似度
        euclidean_similarities = contrast.apply(lambda row: 1 - euclidean(row, current_row_data), axis=1)
        ###########################################################################
        # 找出前 10 相似的行（排除自己）
        top_10_indices = euclidean_similarities.nlargest(11).index[1:]  # 取前 11 名，排除自己
        for top_10 in df.iloc[top_10_indices, -3]:
            output.append(top_10)
        # 檢查這些行的答案是否與當前行相同
        matches = (df.iloc[top_10_indices, -1] == current_answer).sum()  # 第 202 欄（答案）

        # 計算準確率
        accuracy = matches / 10  # 符合的比例
        accuracies.append(accuracy)
        outputs.append(output)
        writer.writerow({'Output': outputs[i], 'Accuracy': accuracies[i]})

data_name = pd.DataFrame(outputs)
data_name.to_csv('formal_eu_data_name.txt', index=False, encoding='utf-8')
# 計算總準確率
overall_accuracy = sum(accuracies) / len(accuracies)
print(f'總準確率: {overall_accuracy:.2%}')

'''
###正規化的準確率###
# 將準確率寫入 txt 檔案
with open('fnormal_accuracy.txt', 'w') as file:
    file.write(f'總準確率: {overall_accuracy:.2%}')
'''

###非正規化的準確率###
# 將準確率寫入 txt 檔案
with open('formal_accuracy.txt', 'a') as file:
    file.write(f' euclidean_similarities 總準確率: {overall_accuracy:.2%}\n')