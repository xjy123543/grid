# -*- coding: utf-8 -*-
"""
基于RFE选择的特征 进行训练模型；
@author: wangxuechao
"""
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from numpy.f2py.crackfortran import verbose
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
)
from sklearn.calibration import calibration_curve

from scipy.stats import chi2
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import ParameterGrid

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']



def specificity_score(y_true, y_pred):
    """计算特异性 (Specificity)。"""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def sensitivity_score(y_true, y_pred):
    """计算敏感性 (Sensitivity, 即 Recall)。"""
    return recall_score(y_true, y_pred)




def lightgbm_train(x_train, y_train, objective, weight_dict, score, parameters):
    x_train, y_train = shuffle(x_train, y_train, random_state=7)
    # 自定义评分指标
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'sensitivity': make_scorer(sensitivity_score),
        'specificity': make_scorer(specificity_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    }
    # 初始化随机森林和交叉验证策略
    model = LGBMClassifier(class_weight=weight_dict, random_state=7, verbose=-1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

    # 设置网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=kfold,
        scoring=scoring,
        refit=score,  # 默认以哪一个指标为最优模型标准
        verbose=3,
        n_jobs=-1
    )

    print("开始网格搜索最优参数...")
    try:
        # 执行网格搜索
        grid_search.fit(x_train, y_train)
        # 打印最优组合和结果
        print(f"最佳参数组合: {grid_search.best_params_}")

    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        return


    # 最优模型
    best_model = grid_search.best_estimator_

    # 创建存储路径
    save_path = os.path.join(f'../checkpoint/{objective}')
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, 'lightgbm_model.pkl')

    # 保存模型
    try:
        joblib.dump(best_model, model_file)
        print(f"模型已保存至: {model_file}")
    except Exception as e:
        print(f"模型存储失败: {str(e)}")

    # 绘制混淆矩阵
    print("绘制最优模型的混淆矩阵...")
    try:
        # 使用训练集预测
        y_pred_train = best_model.predict(x_train)
        print("\n保存的模型在训练集上的表现：")
        print(classification_report(y_train, y_pred_train))
        # 混淆矩阵
        cm = confusion_matrix(y_train, y_pred_train)

        # 绘图
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["未发生", "发生"], yticklabels=["未发生", "发生"])
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        plt.title(f'LightGBM-最优参数组合在训练集上的混淆矩阵 - {objective}')
        plt.show()

    except Exception as e:
        print(f"绘制混淆矩阵失败: {str(e)}")





if __name__ == "__main__":

    warnings.warn("This script is aimed to inter train this model.")

    # 设置目标类型和数据路径
    objective = 'prognosis'  # choices=['prognosis']
    rfe_selected = 'yes'
    data_path = f'../data/dataset/{objective}'


    # 定义超参数搜索空间(****特别注意：为了结果的可复现，不要修改超参数设置)
    if objective == 'prognosis':
        score = 'roc_auc'
        weight_dict = {0:1, 1:1.29}
        threshold = 0.5
        parameters = {
            'learning_rate': [0.1],  # 默认值为0.1
            'num_leaves': [15],  # 默认值为31
            'n_estimators': [100],  # 默认值为100
            'max_depth': [-1],  # 默认值为-1（不限制深度）
            'lambda_l1': [0.0],  # 默认值为0.0
            'lambda_l2': [0.1],  # 默认值为0.0
        }
        selected_features = ['性别-男', '性别-女', '年龄', '发病部位-腮腺', '发病部位-颌下腺',
                             '发病部位-舌下腺+口底', '发病部位-腭', '发病部位-磨牙后区', '发病部位-颊',
                             '发病部位-舌', '发病部位-上颌', '病理类型-鳞状细胞癌', '病理类型-高分化粘表',
                             '病理类型-中分化粘表', '病理类型-低分化粘表', '病理类型-腺样囊性癌',
                             '病理类型-癌在多形性腺瘤中', '病理类型-非特异性腺癌', '病理类型-腺泡细胞癌',
                             '病理类型-肌上皮癌', '病理类型-多型性腺癌', '病理类型-唾液腺导管癌',
                             'T1', 'T2', 'T3', 'T4', 'N0', 'N1', 'N2']



    # 加载数据
    train_file = os.path.join(data_path, 'train.xlsx')
    data_train = pd.read_excel(train_file, engine='openpyxl')

    # 分离特征和标签
    data_feature_train, data_label_train = data_train.iloc[:, :-1].copy(), data_train.iloc[:, -1].copy()
    train_positive_ratio = (data_label_train.sum() / len(data_label_train)) * 100
    print(f"训练集中类别 1 的占比: {train_positive_ratio:.2f}%")

    # 根据选择的特征，筛选出训练数据中相应的特征
    data_feature_train_selected = data_feature_train[selected_features]
    # 根据选择的特征，筛选出数据中相应的特征
    if rfe_selected == 'yes':
        data_feature_train_selected = data_feature_train[selected_features]
    elif rfe_selected == 'no':
        data_feature_train_selected = data_feature_train


    # 模型训练
    lightgbm_train(data_feature_train_selected, data_label_train, objective, weight_dict, score, parameters)



