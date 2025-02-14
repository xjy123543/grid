# -*- coding: utf-8 -*-
"""
网格搜索最优超参数组合；
@author: wangxuechao
"""
import warnings
import sys
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score)
import os                                                                                                                                                                                                                                                                                                                                                                                                                                                 
from sklearn.feature_selection import RFE

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import ParameterGrid
import time

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

start_time = time.time()


# 设置日志保存路径
log_dir = '../log_lgb'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 生成日志文件名
log_file = os.path.join(log_dir, f"rfe+gridsearch-1.25-1.26-1.35-1.3{datetime.now().strftime('%Y-%m-%d_%H')}.log")

# 创建一个自定义的输出类
class Logger:
    def __init__(self, file_path):
        self.console = sys.stdout
        self.file = open(file_path, 'a', encoding='utf-8')

    def write(self, message):
        self.console.write(message)  # 输出到控制台
        self.file.write(message)  # 保存到文件

    def flush(self):
        self.console.flush()
        self.file.flush()

# 重定向标准输出
sys.stdout = Logger(log_file)


def specificity_score(y_true, y_pred):
    """计算特异性 (Specificity)。"""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def sensitivity_score(y_true, y_pred):
    """计算敏感性 (Sensitivity, 即 Recall)。"""
    return recall_score(y_true, y_pred)





def lightgbm_train_custom(
    data_feature_train,
    
    data_label_train,
    data_feature_test,
    data_label_test,
    parameters,
    weight_dict,
):

    best_results = []
    print("  开始网格搜索最优参数，并执行 RFE 特征选择...")

    for param_combo in ParameterGrid(parameters):


        # **步骤 1: 先用 RFE 进行特征选择**
        base_model = LGBMClassifier(**param_combo, class_weight=weight_dict, random_state=7, verbose=-1)
        best_auc = 0
        best_features = None
        for n_features in range(1, data_feature_train.shape[1] + 1, 1):  # 以步长 1 递增
            rfe = RFE(base_model, n_features_to_select=n_features, step=1)
            rfe.fit(data_feature_train, data_label_train)
            selected_features = data_feature_train.columns[rfe.support_]
            # 用选出的特征子集训练模型
            model = LGBMClassifier(**param_combo, class_weight=weight_dict, random_state=7, verbose=-1)
            model.fit(data_feature_train[selected_features], data_label_train)
            # 计算测试集 AUC
            y_train_prob = model.predict_proba(data_feature_train[selected_features])[:, 1]
            auc = roc_auc_score(data_label_train, y_train_prob)
            # 记录 AUC 最优的特征集合
            if auc > best_auc:
                best_auc = auc
                best_features = selected_features
        if best_features is not None:
           print(f"参数组合: {param_combo}")
           print(f"最佳特征组合 ({len(best_features)} 个特征): {list(best_features)}\n")
        else:
           print(f"参数组合: {param_combo} 未找到最佳特征组合\n")



        # **步骤 2: 用最优特征集合重新训练 LightGBM**
        model = LGBMClassifier(**param_combo, class_weight=weight_dict, random_state=7, verbose=-1)
        model.fit(data_feature_train[best_features], data_label_train)
        # 预测训练集和测试集
        y_train_pred = model.predict(data_feature_train[best_features])
        y_train_prob = model.predict_proba(data_feature_train[best_features])[:, 1]
        y_test_pred = model.predict(data_feature_test[best_features])
        y_test_prob = model.predict_proba(data_feature_test[best_features])[:, 1]
        # 计算混淆矩阵
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(data_label_train, y_train_pred).ravel()
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(data_label_test, y_test_pred).ravel()
        # 计算训练集指标
        train_results = {
            "accuracy": accuracy_score(data_label_train, y_train_pred),
            "roc_auc": roc_auc_score(data_label_train, y_train_prob),
            "recall": recall_score(data_label_train, y_train_pred),
            "specificity": train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0,
            "ppv": train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0,
            "npv": train_tn / (train_tn + train_fn) if (train_tn + train_fn) > 0 else 0,
        }
        # 计算测试集指标
        test_results = {
            "accuracy": accuracy_score(data_label_test, y_test_pred),
            "roc_auc": roc_auc_score(data_label_test, y_test_prob),
            "recall": recall_score(data_label_test, y_test_pred),
            "specificity": test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0,
            "ppv": test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0,
            "npv": test_tn / (test_tn + test_fn) if (test_tn + test_fn) > 0 else 0,
        }




        # **步骤 3: 判断超参数组合是否符合筛选标准**
        auc_diff = abs(train_results["roc_auc"] - test_results["roc_auc"])
        if (
                train_results["accuracy"] > test_results["accuracy"]
                and auc_diff <= 0.07
                and test_results['accuracy'] >= 0.75

        ):
            best_results.append(
                {
                    "params": param_combo,
                    "best_features": list(best_features),
                    "train_results": train_results,
                    "test_results": test_results,
                    "weight_dict": weight_dict,
                }
            )

    # **输出符合条件的结果**
    if best_results:
        print("  符合条件的参数组合及其结果：")
        for result in best_results:
            print(f"    参数组合: {result['params']}")
            print(f"    类别权重: {result['weight_dict']}")
            print(f"    选出的特征: {result['best_features']}")
            print(f"    训练集指标: {result['train_results']}")
            print(f"    测试集指标: {result['test_results']}\n")
    else:                                                                                                                                                                                                                                                                                                                                                                                                                 
        print("  没有找到符合条件的参数组合。")

    return best_results








if __name__ == "__main__":

    warnings.warn("This script is aimed to grid search for the optimal hyperparameter combination with RFE.")


    # 设置目标类型和数据路径
    objective = 'prognosis'
    data_path = f'{objective}'


    # 定义超参数搜索空间(****注意：为了结果的可复现，不要修改该超参数设置)
    if objective == 'prognosis':
        threshold = 0.5
        # 44.58% - 1 : 1.24
        weight_dicts = [ {0: 1, 1: 1.25}, {0: 1, 1: 1.26}, {0:1,1:1.35}, {0:1,1:1.3}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                        ]# {0: 1, 1: 1.23}, 'balanced', {0: 1, 1: 1.25}, {0: 1, 1: 1.26},{0: 1, 1: 1.15}, {0: 1, 1: 1.2}, {0: 1, 1: 1.21},
                                        #{0: 1, 1: 1.22}
                        #{0: 1, 1: 27}, {0: 1, 1: 1.28}, {0: 1, 1: 1.29}, {0: 1, 1: 1.3}, {0: 1, 1: 1.35}, {0: 1, 1: 1.3}, {0: 1, 1: 1.35}

        parameters = {
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'num_leaves': [15, 31, 63, 127],
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [-1, 3, 5, 7],
            'lambda_l1': [0, 0.1],
            'lambda_l2': [0, 0.1],
        }


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
    print(f"训练集中类别 1 的占比: {train_positive_ratio:.2f}%")
    print(f"测试集中类别 1 的占比: {test_positive_ratio:.2f}%")


    # 计算总循环次数
    total_iterations = len(weight_dicts)
    iteration_count = 0


    # 模型训练
    for weight_dict in weight_dicts:

        iteration_count += 1
        progress = (iteration_count / total_iterations) * 100
        print(f"\n##当前进度: {progress:.2f}% - 正在进行：weight_dict={weight_dict}")

        lightgbm_train_custom(
            data_feature_train,
            data_label_train,
            data_feature_test,
            data_label_test,
            parameters,
            weight_dict,
        )                                                                          

# 恢复默认输出
sys.stdout.file.close()
sys.stdout = sys.stdout.console

# 记录结束时间并打印运行时长
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n总运行时长: {elapsed_time:.2f} 秒（约 {elapsed_time / 60:.2f} 分钟）")