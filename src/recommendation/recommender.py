"""
志愿推荐引擎模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CollegeRecommender:
    """高考志愿推荐引擎"""

    def __init__(self):
        self.data = None
        self.data_file = None

    def load_data(self, data: pd.DataFrame = None, data_file: str = None):
        """
        加载数据

        Args:
            data: 数据DataFrame
            data_file: 数据文件路径
        """
        if data is not None:
            self.data = data.copy()
        elif data_file is not None:
            self.data = pd.read_csv(data_file, encoding='utf-8-sig')
            self.data_file = data_file
        else:
            # 尝试默认路径
            default_paths = [
                Path(__file__).parent.parent.parent / 'data' / 'processed_college_data.csv',
                Path(__file__).parent.parent.parent / 'data' / 'extracted_tables' / 'admission_data_all.csv',
            ]
            for path in default_paths:
                if path.exists():
                    self.data = pd.read_csv(path, encoding='utf-8-sig')
                    self.data_file = path
                    break

        if self.data is not None:
            # 标准化列名
            self._normalize_columns()
            logger.info(f"推荐引擎加载数据: {len(self.data)} 条记录")
        else:
            logger.warning("未找到数据文件，使用示例数据")
            self._create_sample_data()

    def _normalize_columns(self):
        """标准化列名"""
        if self.data is None:
            return

        # 列名映射
        column_mapping = {
            '院校代码': 'college_code',
            '院校名称': 'college_name',
            '专业组代码': 'major_group',
            '计划数': 'plan_count',
            '投档人数': 'admitted_count',
            '投档最低分': 'min_score',
            '投档最低排位': 'min_rank',
            '年份': 'year',
            '类别': 'category',
        }

        # 重命名存在的列
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.columns:
                rename_dict[old_name] = new_name

        if rename_dict:
            self.data = self.data.rename(columns=rename_dict)

        # 添加缺失的列
        if 'employment_rate' not in self.data.columns:
            self.data['employment_rate'] = self._estimate_employment_rate()

        if 'avg_salary' not in self.data.columns:
            self.data['avg_salary'] = self._estimate_salary()

        if 'rating' not in self.data.columns:
            self.data['rating'] = self._estimate_rating()

        if 'major_name' not in self.data.columns:
            self.data['major_name'] = '待定专业'

        if 'required_subject' not in self.data.columns:
            self.data['required_subject'] = self._estimate_subject_requirement()

    def _estimate_employment_rate(self) -> pd.Series:
        """估算就业率"""
        if 'min_score' in self.data.columns:
            scores = self.data['min_score'].fillna(500)
            return 85 + (scores - 500) / 30 + np.random.uniform(-2, 3, len(self.data))
        return pd.Series([90.0] * len(self.data))

    def _estimate_salary(self) -> pd.Series:
        """估算薪资"""
        if 'min_score' in self.data.columns:
            scores = self.data['min_score'].fillna(500)
            return (7000 + (scores - 500) * 50 + np.random.uniform(-500, 1000, len(self.data))).astype(int)
        return pd.Series([8000] * len(self.data))

    def _estimate_rating(self) -> pd.Series:
        """估算评级"""
        if 'min_score' in self.data.columns:
            scores = self.data['min_score'].fillna(500)
            ratings = []
            for s in scores:
                if s >= 650:
                    ratings.append(np.random.choice(['A+', 'A']))
                elif s >= 600:
                    ratings.append(np.random.choice(['A', 'A-', 'B+']))
                elif s >= 550:
                    ratings.append(np.random.choice(['B+', 'B']))
                elif s >= 500:
                    ratings.append(np.random.choice(['B', 'B-', 'C+']))
                else:
                    ratings.append(np.random.choice(['C+', 'C']))
            return pd.Series(ratings)
        return pd.Series(['B'] * len(self.data))

    def _estimate_subject_requirement(self) -> pd.Series:
        """估算选科要求"""
        if 'category' in self.data.columns:
            requirements = []
            for cat in self.data['category']:
                if '物理' in str(cat):
                    requirements.append('物理')
                elif '历史' in str(cat):
                    requirements.append('历史')
                else:
                    requirements.append('不限')
            return pd.Series(requirements)
        return pd.Series(['不限'] * len(self.data))

    def _create_sample_data(self):
        """创建示例数据"""
        logger.info("创建示例数据...")

        colleges = [
            '中山大学', '华南理工大学', '暨南大学', '华南师范大学', '深圳大学',
            '广州大学', '广东工业大学', '华南农业大学', '广东外语外贸大学',
            '南方医科大学', '广州中医药大学', '汕头大学', '广东财经大学',
            '广东技术师范大学', '仲恺农业工程学院'
        ]

        majors = ['计算机科学与技术', '软件工程', '电子信息工程', '工商管理', '会计学']

        data = []
        np.random.seed(42)

        for college in colleges:
            base_score = 650 - colleges.index(college) * 5

            for major in majors[:3]:
                score = base_score + np.random.normal(0, 3)
                data.append({
                    'college_name': college,
                    'major_name': major,
                    'min_score': round(max(450, min(700, score)), 1),
                    'min_rank': int(50000 / (1 + np.exp((score - 550) / 30))) + 1000,
                    'employment_rate': round(95 - colleges.index(college) * 0.3 + np.random.normal(0, 1), 1),
                    'avg_salary': int(15000 - colleges.index(college) * 300 + np.random.normal(0, 500)),
                    'rating': ['A+', 'A', 'A-', 'B+', 'B'][min(colleges.index(college) // 3, 4)],
                    'required_subject': '物理',
                    'year': 2024,
                    'category': '普通类(物理)'
                })

        self.data = pd.DataFrame(data)

    def filter_by_score(self, candidate_score: int, tolerance: int = 50) -> pd.DataFrame:
        """根据分数筛选"""
        if self.data is None:
            return pd.DataFrame()

        min_score = candidate_score - tolerance
        max_score = candidate_score + tolerance

        filtered = self.data[
            (self.data['min_score'] >= min_score) &
            (self.data['min_score'] <= max_score)
        ]

        return filtered.copy()

    def filter_by_subjects(self, selected_subjects: List[str]) -> pd.DataFrame:
        """根据选科筛选"""
        if self.data is None:
            return pd.DataFrame()

        if selected_subjects is None or len(selected_subjects) == 0:
            return self.data.copy()

        if 'required_subject' not in self.data.columns:
            return self.data.copy()

        def check_match(req):
            if pd.isna(req) or req == '不限' or req == '':
                return True
            req_str = str(req)
            if '和' in req_str:
                required = req_str.split('和')
                return all(r.strip() in selected_subjects for r in required)
            if '或' in req_str:
                required = req_str.split('或')
                return any(r.strip() in selected_subjects for r in required)
            return req_str in selected_subjects

        filtered = self.data[self.data['required_subject'].apply(check_match)]
        return filtered.copy()

    def calculate_probability(self, candidate_rank: int, college_rank: int) -> float:
        """计算录取概率"""
        if college_rank <= 0:
            return 0.5

        ratio = candidate_rank / college_rank

        if ratio <= 0.85:
            return 0.98
        elif ratio <= 0.90:
            return 0.92
        elif ratio <= 0.95:
            return 0.85
        elif ratio <= 1.00:
            return 0.75
        elif ratio <= 1.05:
            return 0.60
        elif ratio <= 1.10:
            return 0.45
        elif ratio <= 1.15:
            return 0.30
        elif ratio <= 1.25:
            return 0.15
        elif ratio <= 1.40:
            return 0.05
        else:
            return 0.02

    def recommend(self,
                  candidate_score: int,
                  candidate_rank: int,
                  selected_subjects: List[str] = None,
                  preference: str = 'balanced',
                  top_n: int = 30) -> Dict:
        """
        生成推荐方案

        Args:
            candidate_score: 考生分数
            candidate_rank: 考生排位
            selected_subjects: 选考科目
            preference: 偏好 (balanced/academic/employment)
            top_n: 返回数量

        Returns:
            推荐方案字典
        """
        if self.data is None:
            logger.warning("数据未加载")
            return {'rush': [], 'stable': [], 'safe': [], 'all': []}

        logger.info(f"生成推荐: 分数={candidate_score}, 排位={candidate_rank}")

        # 筛选数据
        df = self.filter_by_subjects(selected_subjects or [])

        if len(df) == 0:
            logger.warning("没有符合选科要求的数据")
            return {'rush': [], 'stable': [], 'safe': [], 'all': []}

        # 去重
        if 'college_name' in df.columns and 'major_name' in df.columns:
            df = df.drop_duplicates(subset=['college_name', 'major_name'])

        # 计算录取概率
        probabilities = []
        for _, row in df.iterrows():
            min_rank = row.get('min_rank', candidate_rank * 2)
            if pd.isna(min_rank) or min_rank <= 0:
                min_rank = candidate_rank
            prob = self.calculate_probability(candidate_rank, int(min_rank))
            probabilities.append(prob)

        df['admission_probability'] = probabilities

        # 计算综合得分
        df['comprehensive_score'] = (
            df['min_score'].fillna(500) / 7.5 * 0.3 +
            df['employment_rate'].fillna(90) * 0.3 +
            df['avg_salary'].fillna(8000) / 100 * 0.2 +
            df['admission_probability'] * 20
        )

        # 根据偏好调整
        if preference == 'academic':
            df['final_score'] = df['comprehensive_score'] * 0.7 + df['min_score'].fillna(500) / 7.5 * 0.3
        elif preference == 'employment':
            df['final_score'] = df['comprehensive_score'] * 0.5 + df['employment_rate'].fillna(90) * 0.3 + df['avg_salary'].fillna(8000) / 100 * 0.2
        else:
            df['final_score'] = df['comprehensive_score']

        # 排序
        df = df.sort_values('final_score', ascending=False)

        # 分类
        rush, stable, safe = [], [], []

        for _, row in df.iterrows():
            prob = row['admission_probability']

            item = {
                'college': str(row.get('college_name', '')),
                'major': str(row.get('major_name', '')),
                'avg_score': float(row.get('min_score', 0)),
                'min_rank': int(row.get('min_rank', 0)) if pd.notna(row.get('min_rank')) else 0,
                'probability': round(prob * 100, 1),
                'employment_rate': float(row.get('employment_rate', 90)),
                'avg_salary': int(row.get('avg_salary', 8000)),
                'rating': str(row.get('rating', 'B'))
            }

            if prob >= 0.75:
                if len(safe) < 10:
                    safe.append(item)
            elif prob >= 0.55:
                if len(stable) < 10:
                    stable.append(item)
            else:
                if len(rush) < 10:
                    rush.append(item)

        # 合并
        all_items = []
        for item in rush:
            all_items.append({**item, 'type': '冲'})
        for item in stable:
            all_items.append({**item, 'type': '稳'})
        for item in safe:
            all_items.append({**item, 'type': '保'})

        return {
            'rush': rush,
            'stable': stable,
            'safe': safe,
            'all': all_items[:top_n]
        }

    def get_statistics(self) -> Dict:
        """获取数据统计"""
        if self.data is None:
            return {
                'total_colleges': 0,
                'total_majors': 0,
                'avg_score_mean': 0,
                'employment_rate_mean': 90,
                'avg_salary_mean': 8000
            }

        return {
            'total_colleges': self.data['college_name'].nunique() if 'college_name' in self.data.columns else 0,
            'total_majors': len(self.data),
            'avg_score_mean': round(self.data['min_score'].mean(), 1) if 'min_score' in self.data.columns else 0,
            'employment_rate_mean': round(self.data['employment_rate'].mean(), 1) if 'employment_rate' in self.data.columns else 90,
            'avg_salary_mean': int(self.data['avg_salary'].mean()) if 'avg_salary' in self.data.columns else 8000
        }