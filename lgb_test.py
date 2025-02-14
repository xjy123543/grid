# -*- coding: utf-8 -*-
"""
基于RFE选择的特征和训练好的模型结合bootstrap=1000进行测试；
@author: wangxuechao
"""

import warnings
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import numpy as np
import pandas as pd

from utils.calibration_curve import plot_calibration_curve_with_ci
from utils.dca_curve import plot_dca_with_ci
from utils.roc_curve import plot_roc_curve_with_ci
from utils.hl_test import hosmer_lemeshow_test


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def specificity_score(y_true, y_pred):
    """计算特异性 (Specificity)。"""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def sensitivity_score(y_true, y_pred):
    """计算敏感性 (Sensitivity, 即 Recall)。"""
    return recall_score(y_true, y_pred)



def lightgbm_test_with_metrics_ci(x_test, y_test, objective, data_type, threshold=0.5, n_bootstrap=1000, ci=0.95):
    print(f"\n{data_type}数据集上的测试结果:")

    # 加载模型
    lgb = joblib.load(os.path.join(f'../checkpoint/{objective}', 'lightgbm_model.pkl'))
    predict_prob_test = lgb.predict_proba(x_test)[:, 1]  # 获取正类概率
    predict_test = (predict_prob_test >= threshold).astype(int)

    # 初始化存储每次重采样的实际标签和预测概率
    bootstrap_y_test = []
    bootstrap_predict_pro_test  = []
    bootstrap_predict_test = []


    # 初始化用于存储指标的字典
    metrics_dict = {
        "Accuracy": [],
        "Sensitivity": [],
        "Specificity": [],
        "F1-Score": [],
        "AUC": [],
        "PPV": [],
        "NPV": [],
        "Brier Score": [],
        "Log-Loss": [],
        "H-L p-value": []
    }

    for i in range(n_bootstrap):
        # 获取正类和负类的索引
        positive_indices = np.where(y_test == 1)[0]
        negative_indices = np.where(y_test == 0)[0]
        resampled_positive_indices = resample(positive_indices, replace=True, n_samples=len(positive_indices),
                                              random_state=i)
        resampled_negative_indices = resample(negative_indices, replace=True, n_samples=len(negative_indices),
                                              random_state=i)
        resampled_indices = np.concatenate([resampled_positive_indices, resampled_negative_indices])

        # 根据索引生成分层重采样后的数据
        resampled_y_test = y_test[resampled_indices]
        resampled_predict_prob_test = predict_prob_test[resampled_indices]
        resampled_predict_test = predict_test[resampled_indices]

        # 对于分层重采样的数据计算各个区分度指标
        tn, fp, fn, tp = confusion_matrix(resampled_y_test, resampled_predict_test).ravel()
        metrics_dict["Accuracy"].append(accuracy_score(resampled_y_test, resampled_predict_test))
        metrics_dict["Sensitivity"].append(recall_score(resampled_y_test, resampled_predict_test))
        metrics_dict["Specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics_dict["F1-Score"].append(f1_score(resampled_y_test, resampled_predict_test))
        metrics_dict["AUC"].append(roc_auc_score(resampled_y_test, resampled_predict_prob_test))
        metrics_dict["PPV"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics_dict["NPV"].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        metrics_dict["Brier Score"].append(brier_score_loss(resampled_y_test, resampled_predict_prob_test))
        metrics_dict["Log-Loss"].append(log_loss(resampled_y_test, resampled_predict_prob_test))
        metrics_dict["H-L p-value"].append(hosmer_lemeshow_test(resampled_y_test, resampled_predict_prob_test))

        # 收集分层重采样的数据用于后续计算
        bootstrap_y_test.append(resampled_y_test)
        bootstrap_predict_pro_test.append(resampled_predict_prob_test)
        bootstrap_predict_test.append(resampled_predict_test)


    # 计算每个指标的均值
    metrics_mean = {metric: np.mean(values) for metric, values in metrics_dict.items()}

    # 打印结果
    results = {
        "Metric": [
            "Accuracy", "Sensitivity", "Specificity", "F1-Score", "AUC",
            "PPV", "NPV", "Brier Score", "Log-Loss", "H-L p-value"
        ],
        "Value": [
            metrics_mean["Accuracy"], metrics_mean["Sensitivity"], metrics_mean["Specificity"],
            metrics_mean["F1-Score"], metrics_mean["AUC"], metrics_mean["PPV"],
            metrics_mean["NPV"], metrics_mean["Brier Score"], metrics_mean["Log-Loss"],
            metrics_mean["H-L p-value"]
        ],
        "95% CI": [
            f"[{np.percentile(metrics_dict['Accuracy'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['Accuracy'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['Sensitivity'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['Sensitivity'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['Specificity'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['Specificity'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['F1-Score'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['F1-Score'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['AUC'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['AUC'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['PPV'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['PPV'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['NPV'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['NPV'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['Brier Score'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['Brier Score'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['Log-Loss'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['Log-Loss'], (1 + ci) / 2 * 100):.4f}]",
            f"[{np.percentile(metrics_dict['H-L p-value'], (1 - ci) / 2 * 100):.4f}, {np.percentile(metrics_dict['H-L p-value'], (1 + ci) / 2 * 100):.4f}]"

        ]
    }
    results_df = pd.DataFrame(results)
    print(f"\n{threshold} 阈值下模型评估结果及置信区间在{data_type}数据集上：")
    print(results_df)

    # 绘制ROC曲线
    plot_roc_curve_with_ci('LightGBM', bootstrap_y_test, bootstrap_predict_pro_test, ci=0.95)

    # 绘制概率校准度曲线
    plot_calibration_curve_with_ci('LightGBM', bootstrap_y_test, bootstrap_predict_pro_test, ci=0.95)

    # 绘制DCA曲线
    plot_dca_with_ci('LightGBM', bootstrap_y_test, bootstrap_predict_pro_test, ci=0.95)


if __name__ == "__main__":

    warnings.warn("This script is aimed to inter and outside test this model.")


    # 设置目标类型和数据路径
    objective = 'prognosis'
    rfe_selected = 'yes' # choices=['yes', 'no']


    data_path = f'../data/dataset/{objective}'

    # 对应选取的特征名称
    selected_features = ['性别-男', '性别-女', '年龄', '发病部位-腮腺', '发病部位-颌下腺',
                         '发病部位-舌下腺+口底', '发病部位-腭', '发病部位-磨牙后区', '发病部位-颊',
                         '发病部位-舌', '发病部位-上颌', '病理类型-鳞状细胞癌', '病理类型-高分化粘表',
                         '病理类型-中分化粘表', '病理类型-低分化粘表', '病理类型-腺样囊性癌',
                         '病理类型-癌在多形性腺瘤中', '病理类型-非特异性腺癌', '病理类型-腺泡细胞癌',
                         '病理类型-肌上皮癌', '病理类型-多型性腺癌', '病理类型-唾液腺导管癌',
                         'T1', 'T2', 'T3', 'T4', 'N0', 'N1', 'N2']

    # 加载数据
    train_file = os.path.join(data_path, 'train.xlsx')
    test_file = os.path.join(data_path, 'test.xlsx')
    data_train = pd.read_excel(train_file, engine='openpyxl')
    data_test = pd.read_excel(test_file, engine='openpyxl')


    # 分离特征和标签
    data_feature_train, data_label_train = data_train.iloc[:, :-1].copy(), data_train.iloc[:, -1].copy()
    data_feature_test, data_label_test = data_test.iloc[:, :-1].copy(), data_test.iloc[:, -1].copy()
    train_positive_ratio = (data_label_train.sum() / len(data_label_train)) * 100
    test_positive_ratio = (data_label_test.sum() / len(data_label_test)) * 100
    print(f"Train数据集中类别 1 的占比: {train_positive_ratio:.2f}%")
    print(f"Test数据集中类别 1 的占比: {test_positive_ratio:.2f}%")


    # 根据选择的特征，筛选出数据中相应的特征
    if rfe_selected == 'yes':
        data_feature_train_selected = data_feature_train[selected_features]
        data_feature_test_selected = data_feature_test[selected_features]
    elif rfe_selected == 'no':
        data_feature_train_selected = data_feature_train
        data_feature_test_selected = data_feature_test


    # # 模型测试
    lightgbm_test_with_metrics_ci(data_feature_train_selected, data_label_train, objective, data_type='Train')

    lightgbm_test_with_metrics_ci(data_feature_test_selected, data_label_test, objective, data_type='Test')


