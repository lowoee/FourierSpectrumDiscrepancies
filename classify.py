import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from joblib import dump, load

class KNNClassifierTrainer:
    def __init__(self, config_path='fitted_coefficients/metadata.json', 
                 data_path='fitted_coefficients/fitted_coefficients.npz',
                 model_save_path='fitted_coefficients/knn_classifier.joblib'):
        """初始化KNN分类器训练器"""
        self.config_path = config_path
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.metadata = self._load_metadata()
        self.class_names = self._get_class_names()
    
    def _load_metadata(self):
        """加载元数据"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"元数据文件不存在: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _get_class_names(self):
        """获取类别名称列表"""
        known_models = self.metadata['known_models']
        unknown_models = self.metadata['unknown_models']
        class_names = ['real'] + known_models + unknown_models
        return class_names
    
    def load_data(self):
        """加载拟合系数和标签"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        data = np.load(self.data_path)
        train_coeffs = data['train_coeffs']
        train_labels = data['train_labels']
        test_coeffs = data['test_coeffs']
        test_labels = data['test_labels']
        return train_coeffs, train_labels, test_coeffs, test_labels
    
    def preprocess_data(self, train_features, test_features):
        """数据预处理：标准化特征"""
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        return train_features_scaled, test_features_scaled, scaler
    
    def optimize_hyperparameters(self, train_features, train_labels):
        """优化KNN超参数"""
        print("开始优化KNN超参数...")
        knn = KNeighborsClassifier()
        
        param_grid = {
            'n_neighbors': range(1, 31),  # 尝试K=1到30
            'weights': ['uniform', 'distance'],  # 权重方式
            'metric': ['euclidean', 'manhattan', 'minkowski']  # 距离度量
        }
        
        grid_search = GridSearchCV(
            knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(train_features, train_labels)
        
        best_params = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        
        print(f"最佳超参数: {best_params}")
        print(f"交叉验证最佳准确率: {best_accuracy:.4f}")
        return grid_search.best_estimator_
    
    def train_knn(self, train_features, train_labels):
        """使用最佳超参数训练KNN模型"""
        print("训练KNN模型...")
        # 这里直接使用优化后的函数，实际应用中可合并超参数优化和训练
        knn = KNeighborsClassifier(
            n_neighbors=5,  # 示例值，实际应使用优化后的参数
            weights='distance',
            metric='euclidean'
        )
        knn.fit(train_features, train_labels)
        return knn
    
    def evaluate_model(self, model, train_features, train_labels, test_features, test_labels):
        """评估模型性能"""
        # 训练集评估
        train_pred = model.predict(train_features)
        train_accuracy = accuracy_score(train_labels, train_pred)
        
        # 测试集评估
        test_pred = model.predict(test_features)
        test_accuracy = accuracy_score(test_labels, test_pred)
        
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        
        # 生成分类报告
        report = classification_report(
            test_labels, test_pred, target_names=self.class_names
        )
        print("分类报告:\n", report)
        
        # 绘制混淆矩阵
        cm = confusion_matrix(test_labels, test_pred)
        self._plot_confusion_matrix(cm)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report
        }
    
    def _plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('KNN分类混淆矩阵')
        plt.tight_layout()
        plt.savefig('fitted_coefficients/knn_confusion_matrix.png')
        plt.close()
    
    def save_model(self, model):
        """保存训练好的KNN模型"""
        dump(model, self.model_save_path)
        print(f"KNN模型已保存至: {self.model_save_path}")
    
    def train_and_evaluate(self):
        """完整的KNN训练和评估流程"""
        print("开始KNN分类器训练流程...")
        
        # 1. 加载数据
        train_coeffs, train_labels, test_coeffs, test_labels = self.load_data()
        print(f"加载数据完成: 训练集大小={len(train_labels)}, 测试集大小={len(test_labels)}")
        
        # 2. 数据预处理
        train_scaled, test_scaled, scaler = self.preprocess_data(train_coeffs, test_coeffs)
        print("数据预处理完成")
        
        # 3. 超参数优化
        knn_model = self.optimize_hyperparameters(train_scaled, train_labels)
        
        # 4. 评估模型
        evaluation_results = self.evaluate_model(
            knn_model, train_scaled, train_labels, test_scaled, test_labels
        )
        
        # 5. 保存模型
        self.save_model(knn_model)
        
        # 6. 保存评估结果
        self._save_evaluation_results(evaluation_results)
        
        print("KNN分类器训练流程完成")
        return knn_model, evaluation_results
    
    def _save_evaluation_results(self, results):
        """保存评估结果"""
        eval_path = 'fitted_coefficients/knn_evaluation.json'
        results_json = {
            'train_accuracy': float(results['train_accuracy']),
            'test_accuracy': float(results['test_accuracy']),
            'classification_report': results['classification_report']
        }
        with open(eval_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"评估结果已保存至: {eval_path}")

if __name__ == '__main__':
    trainer = KNNClassifierTrainer()
    trainer.train_and_evaluate()