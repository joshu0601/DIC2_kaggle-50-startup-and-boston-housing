# 50 Startups · Multiple Linear Regression Explorer

單頁互動式資料科學儀表板，支援 **5 種回歸模型** 及 **訓練集 5-fold CV / 示範用測試集選模** 兩種模式。

## 快速開始

- **直接使用**：開啟 `index.html`（完整離線版，資料與所有預先計算的模型結果已嵌入）。無須 Python 伺服器。
- **重新計算結果**：安裝 `requirements.txt` 後在本資料夾執行 `python build_startups_dashboard.py`，程式會從 `50_Startups.csv` 計算並更新 `index.html` / `50_Startups_Interactive_Dashboard.html`。
- **本機測試**：`python -m http.server 8080`，瀏覽 `http://localhost:8080/`。
- **GitHub Pages**：將專案中的 `index.html` 上傳到 Repository 根目錄，Settings → Pages → Deploy from a branch → `main` / `/ (root)`。靜態網站不需要 Flask。

## 完整功能清單

| 模組 | 功能 |
| --- | --- |
| Feature Selection | Lasso (L1 Ranking)、Forward SFS、RFE Backward、ElasticNet Ranking、Exhaustive Search；一鍵總比較五種方法 |
| Feature Count | `k=1~5`，即時更新被選擇的特徵、績效、方程式、預測與殘差 |
| Regression Models | Ordinary Least Squares、Ridge、Lasso、ElasticNet、Bayesian Ridge；支援五模型曲線比較 |
| Evaluation | Test RMSE、Test R²、Test adjusted R²、Test MAE、5-fold CV RMSE，另附 OLS training F-statistic、p-value |
| Charts | RMSE vs k、R² vs k、Actual vs Predicted、Standardized Coefficients |
| Equation | 使用原始單位計算截距與回歸係數，依據選取的模型及特徵即時顯示 |
| Statistics | 40 筆訓練資料的 OLS 係數估計、SE、t、p、95% CI、VIF；非 OLS 模型以同特徵集合的獨立 OLS 參考檢定呈現，不假裝是正則化係數的統計檢定 |
| Live Profit Predictor | R&D、Marketing、Administration：數字輸入及滑桿同步；State 下拉選單（California / Florida / New York）；即時計算預測獲利；Reset 功能 |
| Residual Diagnostic | 10 筆測試集完整預測及殘差，依絕對殘差升冪/降冪排序；絕對殘差大於 $10,000 以 High Residual 標記 |
| Other | 50 筆資料檢視、CRISP-DM 流程解說、模型比較及測試預測 CSV 匯出、統計表 CSV 匯出、資料集 CSV、JSON 匯出 |

## 資料與方法

原始 `50_Startups.csv`：50 筆，R&D Spend、Administration、Marketing Spend、State、Profit。對 State 做 One-Hot Encoding，以 California 為 reference，形成 5 個可選預測變數。

使用 80% 訓練集 / 20% 測試集，`random_state=0`。訓練資料的 5-fold CV 使用 `KFold(n_splits=5, shuffle=True, random_state=42)`，數值變數由每個 fold 的 pipeline 內 `StandardScaler` 處理。Forward SFS 與 Exhaustive Search 依訓練 CV 的 OLS RMSE 選特徵；Lasso、ElasticNet、RFE 按訓練資料內學得的係數／排名選特徵。各模型在同一測試集評估。

**注意**：「Test-set optimization (demo only)」會使用測試資料進行特徵選擇，這具有 selection bias，只用於重現課堂範例，正式報告應優先用 `Training 5-fold CV`。只有 50 筆資料、單次 40/10 分割，請勿將結果過度解讀。Adjusted R² 在只有 10 筆測試資料時變異較大。

### OLS 推論與正則化

網頁中預測方程式會反映目前選定的 OLS / Ridge / Lasso / ElasticNet / Bayesian Ridge 模型；統計檢定部分則是將**相同選定特徵**以**訓練資料重新擬合 OLS**，以避免把 OLS 的 p-value 錯當成 Ridge/Lasso 本身的統計推論。VIF 僅描述預測特徵間的共線性，不能用來判斷因果。

### Reproducibility

```bash
python -m pip install -r requirements.txt
python build_startups_dashboard.py
python -m http.server 8080
```

重新建置會計算 31 個非空特徵子集合 × 5 種回歸模型，並儲存 JSON 分析資料；網頁將資料預先嵌入，所以 **HTML 自身可離線操作，但不能在瀏覽器中上傳任意資料重新訓練**。要換資料、alpha 或切分方式，須修改 Python 建置程式後重建。

## 專案檔案

- `index.html`：GitHub Pages 入口 / 完整單檔頁面。
- `50_Startups_Interactive_Dashboard.html`：相同儀表板的獨立 HTML。
- `dashboard_template.html`：HTML + JS 範本，內含 `__DATA_BLOB__` 插入位置。
- `build_startups_dashboard.py`：Python 訓練、評估、OLS inference 與頁面生成。
- `50_Startups.csv`：資料集。
- `startup_dashboard_data.json`：模型資料快取。
- `requirements.txt`：Python 依賴。
- `README.md`：本文件。
