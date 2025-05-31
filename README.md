# 💰中文財經情感分析模型💰

## ❓ 問題陳述

<h4>

隨著社交媒體和網路論壇的興起，投資者對於股票市場的討論日益熱烈。然而，面對海量的財經討論文本，如何快速且準確地判斷市場情緒成為一大挑戰。我們能否利用深度學習技術，自動識別中文財經文本的情感傾向，協助投資者掌握市場脈動？

</h4>

## 📦 資料集詳情

[FinCUGE-Instruction Dataset](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction) <br>
- 專門針對中文金融領域的指令式資料集
- 包含論壇情緒分析任務的標註資料
- 經過篩選後保留高品質的情感分析樣本
- 三分類情感標籤：積極、消極、中性

## 📊 資料詳情 

### 情感標籤分佈:
| 情感類別    | 標籤值 | 說明                                        |
|-------------|--------|---------------------------------------------|
| `積極`      | 0      | 表達正面情緒的財經文本（如看漲、樂觀）     |
| `消極`      | 1      | 表達負面情緒的財經文本（如看跌、悲觀）     |
| `中性`      | 2      | 表達中性態度的財經文本（如客觀分析）       |

### 資料預處理:
- 使用正則表達式從自然語言輸出中提取情感標籤
- 結合指令文本與輸入文本進行訓練
- 文本截斷至最大長度128 tokens
- 採用BERT中文分詞器進行tokenization

---

## 🧠 模型架構與技術選擇

本專案採用基於BERT的序列分類模型，專門針對中文財經文本進行情感分析。

---

### 🔍 為什麼選擇BERT？

- **預訓練優勢**：bert-base-chinese在大量中文語料上預訓練，具備強大的中文語言理解能力
- **雙向編碼**：能夠同時考慮上下文信息，提升情感理解準確性
- **遷移學習**：在預訓練基礎上微調，有效利用已學習的語言表示
- **序列分類**：直接支援文本分類任務，無需額外架構設計

---

### 🎯 模型配置

- **基礎模型**：bert-base-chinese
- **分類頭**：3類輸出（積極、消極、中性）
- **最大序列長度**：128 tokens
- **優化器**：AdamW
- **學習率**：1e-5
- **批次大小**：16

---

### 📘 訓練策略

- **訓練輪數**：10 epochs
- **早停機制**：連續3個epoch無改善時停止
- **學習率預熱**：500 warmup steps
- **權重衰減**：0.01
- **模型保存**：保留驗證效果最佳的模型

---

## 📈 模型表現評估

### ✅ 評估指標
- **準確率 (Accuracy)**：整體分類正確率
- **F1分數 (Macro F1)**：平衡各類別的F1分數
- **混淆矩陣**：詳細分析各情感類別的分類效果

### 📊 實驗結果
根據實驗結果，模型在測試集上達到約 **78%** 的分類準確率，在中文財經情感分析任務中表現良好。

![模型訓練過程](./results/training_progress.png)
![混淆矩陣](./results/confusion_matrix.png)

---

## 🧪 模型推論範例

### 💡 使用方式

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 載入模型和tokenizer
model_path = "./results/model_20241201_143022"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 情感預測
text = "這支股票今天開低走高，明天應該會繼續漲！"
input_text = "这个文本的情感倾向是积极、消极还是中性的。 " + text

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
# 輸出：積極
```

### 🔍 預測結果解讀
- **積極情感**：通常包含看漲、買入、樂觀等詞彙
- **消極情感**：通常包含看跌、賣出、悲觀等詞彙  
- **中性情感**：通常為客觀分析或觀望態度

---

## 🚀 應用場景

### 💼 主要應用領域
- **股票論壇監控**：自動分析投資社群的情緒趨勢
- **財經新聞分析**：評估新聞報導對市場情緒的影響
- **投資決策輔助**：作為量化交易策略的參考指標
- **風險管理**：識別市場恐慌情緒，提供預警機制

### 📱 部署建議
- **API服務**：可封裝為REST API供其他系統調用
- **批量處理**：支援大規模文本的批量情感分析
- **實時監控**：結合爬蟲技術進行實時情感監控

---

## 🧾 技術優勢與限制

### ✨ 技術優勢
- **領域專化**：專門針對中文財經文本優化
- **高準確性**：達到78%的分類準確率
- **可解釋性**：基於attention機制，可視化重要詞彙
- **易部署**：標準Transformer架構，支援多種部署方式

### ⚠️ 現有限制
- **資料依賴**：需要高品質的標註資料進行訓練
- **領域限制**：主要適用於財經領域，其他領域需重新訓練
- **計算資源**：推論需要一定的GPU資源支援

---

## 🔮 未來改進方向

- **模型升級**：嘗試RoBERTa、ELECTRA等更先進的預訓練模型
- **多模態融合**：結合數值指標（如股價、交易量）進行綜合分析
- **實時系統**：建立完整的實時情感監控系統
- **跨領域擴展**：擴展至其他金融子領域（如債券、期貨）

---

## 🛠️ 安裝與使用

### 環境需求
```bash
pip install torch transformers datasets sentencepiece accelerate sklearn
```

### 快速開始
```bash
git clone https://github.com/yehdanny/Sentiment-Analysis.git
cd Sentiment-Analysis
python 中文財經情感分析.py
```

---

## 📞 聯絡資訊

### Contact Me : 
- [GitHub@yehdanny](https://github.com/yehdanny)
- [Website@yehdanny](https://yehdanny.github.io/mypage/html/index.html)
- [Instagram@yeh_const](https://www.instagram.com/yeh_const?igsh=MTVlNTl2eGVkeWI2MA%3D%3D&utm_source=qr)

---

## 📄 授權條款

本專案採用 MIT 授權條款，詳細內容請參閱 [LICENSE](LICENSE) 檔案。

---

*最後更新：2024年12月*
