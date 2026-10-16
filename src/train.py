import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Ngưỡng chất lượng của lab này là f1_score, KHÔNG phải accuracy.
# Lý do: bộ dữ liệu Adult có tỷ lệ lớp 75/25. Một mô hình đoán bừa
# "thu nhập thấp" cho mọi mẫu đã đạt accuracy 0.75 mà không học được gì.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params     : dict chứa các siêu tham số cho GradientBoostingClassifier.
        data_path  : đường dẫn đến file dữ liệu huấn luyện.
        eval_path  : đường dẫn đến file dữ liệu đánh giá (holdout).

    Trả về:
        f1 (float): điểm F1 của lớp dương (thu nhập > 50K) trên tập holdout.
    """

    # TODO 1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tách đặc trưng (X) và nhãn (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiểm tra lệch lạc phân phối dữ liệu (Data Drift Warning)
    pos_ratio = float(y_train.mean())
    if abs(pos_ratio - REFERENCE_POSITIVE_RATIO) > 0.05:
        print(f"[DATA DRIFT WARNING] Ty le lop duong: {pos_ratio:.1%} (lech > 5% so voi moc 24.8%)")
    else:
        print(f"[DATA CHECK] Ty le lop duong dat chuan: {pos_ratio:.1%} (tham chieu 24.8%)")

    with mlflow.start_run():

        # TODO 3: Ghi nhận các siêu tham số
        mlflow.log_params(params)

        # TODO 4: Khởi tạo và huấn luyện GradientBoostingClassifier
        # Sử dụng random_state=42 để đảm bảo tính tái tạo
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Dự đoán trên tập holdout và tính chỉ số
        # Chú ý: f1_score ở đây tính cho LỚP DƯƠNG (target = 1), không dùng average.
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds, zero_division=0))
        acc = float(accuracy_score(y_eval, preds))

        # Bonus 2: Quét ngưỡng quyết định tối ưu F1
        best_threshold = 0.5
        best_f1 = f1
        try:
            if hasattr(model, "predict_proba") and len(np.unique(y_eval)) > 1:
                y_probs = model.predict_proba(X_eval)[:, 1]
                thresholds = np.arange(0.1, 0.95, 0.05)
                for th in thresholds:
                    th_preds = (y_probs >= th).astype(int)
                    th_f1 = float(f1_score(y_eval, th_preds, zero_division=0))
                    if th_f1 > best_f1:
                        best_f1 = th_f1
                        best_threshold = float(round(th, 2))
        except Exception as e:
            print(f"Bỏ qua tối ưu ngưỡng: {e}")

        # Bonus 3: Báo cáo Precision / Recall chi tiết và Confusion Matrix
        try:
            prec_pos = float(precision_score(y_eval, preds, pos_label=1, zero_division=0))
            rec_pos = float(recall_score(y_eval, preds, pos_label=1, zero_division=0))
            prec_neg = float(precision_score(y_eval, preds, pos_label=0, zero_division=0))
            rec_neg = float(recall_score(y_eval, preds, pos_label=0, zero_division=0))
            cm = confusion_matrix(y_eval, preds)

            os.makedirs("outputs", exist_ok=True)
            with open("outputs/detail.txt", "w", encoding="utf-8") as f_det:
                f_det.write("=== CHI TIẾT ĐÁNH GIÁ MÔ HÌNH (PRECISION / RECALL & CONFUSION MATRIX) ===\n\n")
                f_det.write(f"Confusion Matrix:\n{cm}\n\n")
                f_det.write(f"Lớp 0 (Thu nhập <= 50K) - Precision: {prec_neg:.4f} | Recall: {rec_neg:.4f}\n")
                f_det.write(f"Lớp 1 (Thu nhập > 50K)  - Precision: {prec_pos:.4f} | Recall: {rec_pos:.4f}\n\n")
                f_det.write(f"Accuracy chung: {acc:.4f}\n")
                f_det.write(f"F1-score lớp dương (ngưỡng 0.5): {f1:.4f}\n")
                f_det.write(f"Ngưỡng tối ưu: {best_threshold:.2f} mang lại F1-score: {best_f1:.4f}\n")
        except Exception as e:
            print(f"Bỏ qua ghi nhận chi tiết: {e}")

        # TODO 6: Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_metric("positive_ratio", pos_ratio)
        try:
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
        except Exception:
            mlflow.sklearn.log_model(model, "model")

        # TODO 7: In kết quả ra màn hình
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f} | Ngưỡng tối ưu {best_threshold:.2f} -> Best F1: {best_f1:.4f}")

        # TODO 8: Lưu metrics ra file outputs/report.json
        # File này được đọc bởi GitHub Actions ở Bước 2
        os.makedirs("outputs", exist_ok=True)
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_threshold,
            "best_f1_score": best_f1,
            "positive_ratio": pos_ratio,
        }
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # TODO 9: Lưu mô hình ra file models/model.joblib
        # File này được upload lên cloud storage ở Bước 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Trả về f1
    return f1


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train(params)
