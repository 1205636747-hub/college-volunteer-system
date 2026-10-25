"""
录取概率预测模块 - 基于机器学习的动态概率预测
支持2022-2025多年份数据训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AdmissionPredictor:
    """录取概率预测器 - 支持多年份数据"""

    def __init__(self):
        # 随机森林模型
        self.rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 梯度提升模型
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )

        # 神经网络模型
        self.nn_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_score = 0.0

        # 自动定位项目根目录和模型目录
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent
        self.model_dir = self.project_root / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"模型保存目录: {self.model_dir.absolute()}")

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None,
              years: np.ndarray = None) -> Dict:
        """
        训练多个模型并选择最优

        Args:
            X: 特征矩阵
            y: 目标变量（录取排位）
            feature_names: 特征名称列表
            years: 年份数据（用于分层采样，确保测试集包含各年份数据）

        Returns:
            各模型评估结果
        """
        logger.info("=" * 60)
        logger.info("开始训练录取预测模型（多年份数据）")
        logger.info(f"训练数据: {X.shape[0]} 条记录, {X.shape[1]} 个特征")

        if feature_names:
            self.feature_names = feature_names
            logger.info(f"特征列表: {feature_names}")

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 划分训练集和测试集
        if years is not None and len(np.unique(years)) > 1:
            try:
                year_bins = pd.cut(years, bins=3, labels=False)
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, test_idx = next(sss.split(X_scaled, year_bins))
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                logger.info(f"按年份分层划分: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
            except Exception as e:
                logger.warning(f"分层划分失败，使用随机划分: {e}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            logger.info(f"随机划分: 训练集 {len(X_train)}, 测试集 {len(X_test)}")

        models = {
            'RandomForest': self.rf_model,
            'GradientBoosting': self.gb_model,
            'NeuralNetwork': self.nn_model
        }

        results = {}
        best_score = float('inf')

        for name, model in models.items():
            logger.info(f"\n训练 {name} 模型...")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                relative_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                try:
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5,
                                                scoring='neg_mean_absolute_error')
                    cv_mae = -cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as e:
                    logger.warning(f"交叉验证失败: {e}")
                    cv_mae = mae * 1.1
                    cv_std = 0

                results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Relative_Error': relative_error,
                    'CV_MAE': cv_mae,
                    'CV_Std': cv_std,
                    'model': model
                }

                logger.info(f"  MAE (平均绝对误差): {mae:.2f}")
                logger.info(f"  RMSE (均方根误差): {rmse:.2f}")
                logger.info(f"  R² Score: {r2:.4f}")
                logger.info(f"  相对误差: {relative_error:.2f}%")
                logger.info(f"  CV MAE: {cv_mae:.2f} (±{cv_std:.2f})")

                score = cv_mae * (1 - min(r2, 0.9) * 0.5)
                if score < best_score:
                    best_score = score
                    self.best_model = model
                    self.best_model_name = name
                    self.training_score = r2

            except Exception as e:
                logger.error(f"  {name} 训练失败: {e}")
                import traceback
                traceback.print_exc()

        if self.best_model is not None:
            self.is_trained = True
            logger.info(f"\n{'=' * 60}")
            logger.info(f"✅ 最优模型: {self.best_model_name}")
            logger.info(f"   R² Score: {results[self.best_model_name]['R2']:.4f}")
            logger.info(f"   CV MAE: {results[self.best_model_name]['CV_MAE']:.2f}")
            logger.info(f"   相对误差: {results[self.best_model_name]['Relative_Error']:.2f}%")
            logger.info(f"{'=' * 60}")
        else:
            logger.error("所有模型训练失败")

        return results

    def predict_rank(self, features: np.ndarray) -> np.ndarray:
        """预测录取排位（返回原始排位值）"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")

        features_scaled = self.scaler.transform(features)
        predictions_log = self.best_model.predict(features_scaled)
        # 反变换：从对数空间转回原始排位
        predictions = np.expm1(predictions_log)
        return np.maximum(1, predictions.astype(int))

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            logger.warning("模型未训练，无法预测")
            return df

        result_df = df.copy()
        available_features = [f for f in self.feature_names if f in result_df.columns]

        if len(available_features) < 2:
            logger.warning("特征不足，无法预测")
            result_df['predicted_rank'] = result_df.get('min_rank', 10000)
            return result_df

        for col in available_features:
            if col in result_df.columns:
                median_val = result_df[col].median() if not result_df[col].isna().all() else 0
                result_df[col] = result_df[col].fillna(median_val)

        X = result_df[available_features].fillna(0).values

        try:
            result_df['predicted_rank'] = self.predict_rank(X)
        except Exception as e:
            logger.error(f"预测失败: {e}")
            result_df['predicted_rank'] = result_df.get('min_rank', 10000)

        return result_df

    def calculate_admission_probability(self,
                                        predicted_rank: float,
                                        candidate_rank: int,
                                        historical_ranks: List[float] = None,
                                        score_trend: float = 0) -> float:
        if predicted_rank <= 0:
            return 0.5

        ratio = candidate_rank / predicted_rank

        if ratio <= 0.85:
            base_prob = 0.98
        elif ratio <= 0.90:
            base_prob = 0.92
        elif ratio <= 0.95:
            base_prob = 0.85
        elif ratio <= 1.0:
            base_prob = 0.75
        elif ratio <= 1.05:
            base_prob = 0.60
        elif ratio <= 1.10:
            base_prob = 0.45
        elif ratio <= 1.15:
            base_prob = 0.30
        elif ratio <= 1.25:
            base_prob = 0.15
        elif ratio <= 1.40:
            base_prob = 0.05
        else:
            base_prob = 0.01

        if historical_ranks and len(historical_ranks) > 1:
            std_rank = np.std(historical_ranks)
            mean_rank = np.mean(historical_ranks)
            if mean_rank > 0:
                cv = std_rank / mean_rank
                adjustment = min(0.15, cv * 0.2)
                if base_prob > 0.5:
                    base_prob = base_prob * (1 - adjustment)
                else:
                    base_prob = base_prob * (1 + adjustment)

        if score_trend > 2:
            base_prob = base_prob * 0.95
        elif score_trend < -2:
            base_prob = base_prob * 1.05

        return np.clip(base_prob, 0.01, 0.99)

    def predict_admission_probability(self,
                                      college_data: pd.DataFrame,
                                      candidate_score: int,
                                      candidate_rank: int) -> pd.DataFrame:
        logger.info(f"正在预测录取概率 (考生分数: {candidate_score}, 排位: {candidate_rank})...")

        result_df = college_data.copy()

        if not self.is_trained:
            logger.warning("模型未训练，使用启发式方法")
            return self._heuristic_probability(result_df, candidate_score, candidate_rank)

        result_df = self.predict_batch(result_df)

        probabilities = []
        for _, row in result_df.iterrows():
            predicted_rank = row.get('predicted_rank', row.get('min_rank', candidate_rank * 2))
            score_trend = row.get('score_trend', 0)

            historical = []
            if 'min_rank' in row:
                historical.append(row['min_rank'])

            prob = self.calculate_admission_probability(
                predicted_rank,
                candidate_rank,
                historical if historical else None,
                score_trend
            )
            probabilities.append(prob)

        result_df['admission_probability'] = probabilities

        if 'min_score' in result_df.columns:
            result_df['score_gap'] = candidate_score - result_df['min_score']

        result_df = result_df.sort_values('admission_probability', ascending=False)
        result_df['recommend_category'] = result_df['admission_probability'].apply(
            lambda p: '保' if p >= 0.75 else ('稳' if p >= 0.55 else '冲')
        )

        return result_df

    def _heuristic_probability(self, college_data, candidate_score, candidate_rank):
        """启发式方法 - 优化版"""
        result_df = college_data.copy()
        probabilities = []

        for _, row in result_df.iterrows():
            min_score = row.get('min_score', 500)
            min_rank = row.get('min_rank', candidate_rank * 2)

            if pd.isna(min_rank) or min_rank <= 0:
                min_rank = candidate_rank

            # 计算排位比率
            rank_ratio = candidate_rank / min_rank
            # 计算分数差距
            score_gap = candidate_score - min_score

            # 综合概率计算
            if rank_ratio <= 0.85 and score_gap >= 15:
                prob = 0.95
            elif rank_ratio <= 0.90 and score_gap >= 10:
                prob = 0.88
            elif rank_ratio <= 0.95 and score_gap >= 5:
                prob = 0.80
            elif rank_ratio <= 1.00:
                prob = 0.70
            elif rank_ratio <= 1.05:
                prob = 0.55
            elif rank_ratio <= 1.10:
                prob = 0.40
            elif rank_ratio <= 1.15:
                prob = 0.25
            elif rank_ratio <= 1.25:
                prob = 0.12
            elif rank_ratio <= 1.40:
                prob = 0.05
            else:
                prob = 0.02

            probabilities.append(prob)

        result_df['admission_probability'] = probabilities
        result_df['recommend_category'] = result_df['admission_probability'].apply(
            lambda p: '保' if p >= 0.75 else ('稳' if p >= 0.55 else '冲')
        )

        return result_df.sort_values('admission_probability', ascending=False)

    def save_model(self, filename: str = 'admission_predictor.pkl') -> bool:
        if not self.is_trained:
            logger.warning("模型尚未训练，无法保存")
            return False

        model_path = self.model_dir / filename

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'training_score': self.training_score,
            'is_trained': True
        }

        try:
            joblib.dump(model_data, model_path)
            logger.info(f"✅ 模型已保存至: {model_path.absolute()}")
            return True
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False

    def load_model(self, filename: str = 'admission_predictor.pkl') -> bool:
        model_path = self.model_dir / filename

        if not model_path.exists():
            logger.info(f"模型文件不存在: {model_path}")
            return False

        try:
            model_data = joblib.load(model_path)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.best_model_name = model_data.get('model_name', 'Unknown')
            self.feature_names = model_data.get('feature_names', [])
            self.training_score = model_data.get('training_score', 0)
            self.is_trained = model_data.get('is_trained', True)
            logger.info(f"✅ 模型已加载: {self.best_model_name} (训练R²: {self.training_score:.4f})")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def get_model_info(self) -> Dict:
        return {
            'is_trained': self.is_trained,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'training_score': self.training_score,
            'model_dir': str(self.model_dir.absolute())
        }


if __name__ == '__main__':
    print("=" * 60)
    print("录取概率预测器测试")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 500

    X = np.random.rand(n_samples, 4) * 100
    X[:, 0] = np.random.randint(450, 700, n_samples)
    X[:, 1] = np.random.randint(85, 99, n_samples)
    X[:, 2] = np.random.randint(6000, 20000, n_samples)
    X[:, 3] = np.random.randint(70, 100, n_samples)

    y = 50000 / (1 + np.exp((X[:, 0] - 550) / 30)) + np.random.randint(-500, 500, n_samples)
    y = np.maximum(1, y.astype(int))

    years = np.random.choice([2022, 2023, 2024, 2025], n_samples)

    print(f"\n测试数据: {n_samples} 条")
    print(f"年份分布: {np.unique(years, return_counts=True)}")

    predictor = AdmissionPredictor()
    results = predictor.train(X, y,
                              feature_names=['score', 'employment', 'salary', 'rating'],
                              years=years)

    if predictor.is_trained:
        predictor.save_model()

        print("\n【测试预测】")
        test_candidates = [
            {'score': 650, 'rank': 5000},
            {'score': 580, 'rank': 20000},
            {'score': 500, 'rank': 80000},
        ]

        test_df = pd.DataFrame({
            'college_name': ['测试大学A', '测试大学B', '测试大学C'] * 3,
            'score': [650, 580, 500] * 3,
            'employment': [95, 90, 85] * 3,
            'salary': [15000, 10000, 7000] * 3,
            'rating': [95, 80, 70] * 3,
            'min_rank': [4000, 18000, 75000] * 3
        })

        for candidate in test_candidates:
            result = predictor.predict_admission_probability(
                test_df.head(10),
                candidate['score'],
                candidate['rank']
            )
            print(f"\n考生: 分数={candidate['score']}, 排位={candidate['rank']}")
            for _, row in result.head(3).iterrows():
                prob = row['admission_probability'] * 100
                cat = row['recommend_category']
                print(f"  {row['college_name']}: {prob:.1f}% [{cat}]")