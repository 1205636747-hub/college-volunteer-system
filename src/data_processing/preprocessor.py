"""
数据处理模块 - 数据清洗、归一化、多源数据融合
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理类 - 支持多年份数据"""

    def __init__(self):
        self.scaler = MinMaxScaler()

        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent
        self.data_dir = self.project_root / 'data'
        self.parsed_dir = self.data_dir / 'parsed'
        self.extracted_dir = self.data_dir / 'extracted_tables'
        self.data_dir.mkdir(exist_ok=True)

        logger.info(f"数据目录: {self.data_dir.absolute()}")

        # 构建标准院校名称库
        self.standard_colleges = self._build_standard_college_list()

    def _build_standard_college_list(self) -> set:
        """构建标准院校名称集合"""
        colleges = set()

        # 985院校
        colleges.update([
            '清华大学', '北京大学', '复旦大学', '上海交通大学', '浙江大学',
            '中国科学技术大学', '南京大学', '西安交通大学', '哈尔滨工业大学',
            '武汉大学', '华中科技大学', '中山大学', '华南理工大学',
            '四川大学', '电子科技大学', '重庆大学', '东南大学', '同济大学',
            '北京航空航天大学', '北京理工大学', '中国人民大学', '厦门大学',
            '山东大学', '中国海洋大学', '吉林大学', '大连理工大学', '东北大学',
            '兰州大学', '西北工业大学', '天津大学', '南开大学', '中南大学',
            '湖南大学', '国防科技大学', '西北农林科技大学', '中央民族大学',
            '华东师范大学', '北京师范大学', '东北大学'
        ])

        # 211院校
        colleges.update([
            '上海财经大学', '中央财经大学', '对外经济贸易大学', '北京外国语大学',
            '上海外国语大学', '中国政法大学', '北京邮电大学', '北京交通大学',
            '北京科技大学', '北京化工大学', '北京工业大学', '北京林业大学',
            '北京中医药大学', '中国传媒大学', '中国地质大学', '中国石油大学',
            '中国矿业大学', '华北电力大学', '河海大学', '江南大学', '东华大学',
            '华东理工大学', '上海大学', '苏州大学', '南京航空航天大学',
            '南京理工大学', '南京师范大学', '南京农业大学', '中国药科大学',
            '合肥工业大学', '安徽大学', '福州大学', '南昌大学', '郑州大学',
            '武汉理工大学', '华中农业大学', '华中师范大学', '中南财经政法大学',
            '湖南师范大学', '暨南大学', '华南师范大学', '广西大学', '海南大学',
            '西南交通大学', '西南财经大学', '四川农业大学', '贵州大学',
            '云南大学', '西藏大学', '西北大学', '西安电子科技大学', '长安大学',
            '陕西师范大学', '青海大学', '宁夏大学', '新疆大学', '石河子大学',
            '第二军医大学', '第四军医大学'
        ])

        # 广东省主要院校
        colleges.update([
            '深圳大学', '广州大学', '广东工业大学', '华南农业大学',
            '广东外语外贸大学', '南方医科大学', '广州医科大学',
            '广东财经大学', '广东金融学院', '广东技术师范大学',
            '仲恺农业工程学院', '广东海洋大学', '广东医科大学',
            '广东药科大学', '广州中医药大学', '汕头大学',
            '东莞理工学院', '佛山大学', '五邑大学', '肇庆学院',
            '惠州学院', '韶关学院', '嘉应学院', '韩山师范学院',
            '岭南师范学院', '广东第二师范学院', '广州航海学院',
            '广东石油化工学院', '北京理工大学珠海学院',
            '吉林大学珠海学院', '电子科技大学中山学院',
            '广州南方学院', '广州华商学院', '广州理工学院',
            '广州软件学院', '广州工商学院', '广东白云学院'
        ])

        # 常见师范院校
        colleges.update([
            '北京师范大学', '华东师范大学', '华中师范大学', '东北师范大学',
            '西南大学', '陕西师范大学', '华南师范大学', '湖南师范大学',
            '南京师范大学', '首都师范大学', '福建师范大学', '浙江师范大学',
            '山东师范大学', '上海师范大学', '江西师范大学', '河北师范大学',
            '河南师范大学', '江苏师范大学', '四川师范大学', '安徽师范大学',
            '广西师范大学', '天津师范大学', '辽宁师范大学', '哈尔滨师范大学',
            '曲阜师范大学', '山西师范大学', '吉林师范大学', '沈阳师范大学',
            '杭州师范大学', '重庆师范大学', '贵州师范大学', '云南师范大学',
            '西北师范大学', '青海师范大学', '新疆师范大学', '内蒙古师范大学'
        ])

        # 常见医药院校
        colleges.update([
            '北京协和医学院', '首都医科大学', '天津医科大学', '南京医科大学',
            '重庆医科大学', '哈尔滨医科大学', '中国医科大学', '广州医科大学',
            '南方医科大学', '广东医科大学', '广东药科大学', '广州中医药大学',
            '北京中医药大学', '上海中医药大学', '南京中医药大学',
            '成都中医药大学', '天津中医药大学', '黑龙江中医药大学',
            '辽宁中医药大学', '浙江中医药大学', '山东中医药大学',
            '湖北中医药大学', '湖南中医药大学', '福建中医药大学',
            '江西中医药大学', '安徽中医药大学', '广西中医药大学'
        ])

        return colleges

    def clean_college_name(self, name: str) -> str:
        """
        彻底清理院校名称（暴力去前缀版）
        """
        if pd.isna(name) or name == '' or name == '待定':
            return ''

        original = str(name).strip()
        name = original

        # ========== 强力去括号 ==========
        name = re.sub(r'[（(].*?[）)]', '', name).strip()

        # ========== 第一轮：暴力去除常见解析垃圾前缀 ==========
        # 注意：只删除确定是垃圾的单字/双字，且后面跟着的是真正的院校名开头
        garbage_prefixes = [
            '教', '试', '考', '育', '省', '市', '院',
            '教深圳', '试绍兴', '考上饶', '育韩山', '广肇庆',
            '省湖南', '省新疆', '省甘肃', '省白城', '省伊犁', '省广西', '省宿迁', '省河南'
        ]

        for prefix in garbage_prefixes:
            if name.startswith(prefix):
                candidate = name[len(prefix):]
                # 确保去掉前缀后得到的是有效名称开头
                if len(candidate) >= 3 and (candidate[0] in '北上广深杭武成重西天南青苏'):
                    name = candidate
                    break
                # 如果前缀本身就是完整匹配的乱码+部分名称
                if '大学' in candidate or '学院' in candidate:
                    name = candidate
                    break

        # ========== 第二轮：处理"东"字前缀（精准区分） ==========
        # 只有以下情况才删除开头的"东"字：
        # 1. "东东北" -> "东北"（重复前缀）
        # 2. "东中国" -> "中国"
        # 3. "东北京" -> "北京"
        # 绝不影响"东南大学"、"东北大学"、"东华大学"等合法院校
        if name.startswith('东'):
            if name.startswith('东东北'):
                name = name[1:]
            elif name.startswith('东中国'):
                name = name[1:]
            elif name.startswith('东北京'):
                name = name[1:]
            elif name.startswith('东上海'):
                name = name[1:]
            elif name.startswith('东广州'):
                name = name[1:]
            elif name.startswith('东成都'):
                name = name[1:]
            elif name.startswith('东西安'):
                name = name[1:]
            elif name.startswith('东武汉'):
                name = name[1:]
            elif name.startswith('东安徽'):
                name = name[1:]
            elif name.startswith('东齐鲁'):
                name = name[1:]
            elif name.startswith('东山东'):
                name = name[1:]
            elif name.startswith('东嘉应'):
                name = name[1:]
            elif name.startswith('东新疆'):
                name = name[1:]
            elif name.startswith('东湖南'):
                name = name[1:]
            elif name.startswith('东南'):
                name = name[1:] if name[1:] != '南大学' else name

        # ========== 第三轮：处理"广"字前缀（精准区分） ==========
        # 只删除确定是乱码的情况，保留"广东"、"广州"、"广西"等合法名称
        if name.startswith('广'):
            # 乱码情况：广 + 非广东/广州/广西的地名或院校
            garbage_after_guang = [
                '井冈山', '广州', '广东', '广西', '上海', '北京', '深圳', '成都', '重庆', '武汉', '西安', '南京',
                '天津', '杭州', '苏州', '厦门', '青岛', '大连', '沈阳', '长春',
                '哈尔滨', '济南', '郑州', '合肥', '南昌', '福州', '贵阳', '昆明',
                '兰州', '太原', '石家庄', '五邑', '肇庆', '韶关', '惠州', '梅州',
                '潮州', '揭阳', '汕尾', '河源', '清远', '茂名', '湛江', '阳江',
                '云浮', '东莞', '中山', '珠海', '江门', '佛山', '北'
            ]
            for suffix in garbage_after_guang:
                if name.startswith('广' + suffix):
                    name = name[1:]
                    break
            # 特殊：广广东 -> 广东（去重）
            if name.startswith('广广东'):
                name = name[1:]
            # 特殊：广广州 -> 广州（去重）
            if name.startswith('广广州'):
                name = name[1:]
            # 特殊：广广西 -> 广西（去重）
            if name.startswith('广广西'):
                name = name[1:]
            if name.startswith('广中国'):
                name = name[1:]
            if name.startswith('广首都'):
                name = name[1:]

        # 情况：北师范大学 -> 北京师范大学
        if name.startswith('北') and not name.startswith('北京'):
            if name[1:] in ['京师范大学', '京大学', '京电影学院', '京交通大学',
                           '京航空航天大学', '京理工大学', '京科技大学', '京邮电大学',
                           '京化工大学', '京工业大学', '京林业大学', '京中医药大学',
                           '京体育大学', '京外国语大学', '京语言大学']:
                # 实际上是缺失了"京"字前面的"北"？不，这里是"北"后面直接跟了"师范大学"
                # 正确应该是"北京师范大学"
                pass  # 会在exact_fixes中处理

        # ========== 第四轮：连续重复汉字去重 ==========
        name = re.sub(r'^([\u4e00-\u9fa5])\1+([\u4e00-\u9fa5])', r'\1\2', name)

        # ========== 第五轮：错误双字前缀修正 ==========
        prefix_fixes = {
            '南北北京': '北京', '东西安': '西安', '东西南': '西南', '兰广广州': '广州',
            '广上海': '上海', '广华中': '华中', '广南宁': '南宁', '广广东': '广东','广广西': '广西','广琼台': '琼台', '东中国': '中国', '上北京': '北京', '北上海': '上海',
            '南北京': '北京', '西上海': '上海', '北北北京': '北京', '上上上海': '上海',
            '广广广州': '广州', '深深圳': '深圳', '武武汉': '武汉', '南南京': '南京',
        }
        for wrong, right in prefix_fixes.items():
            if name.startswith(wrong):
                name = right + name[len(wrong):]
                break

        # ========== 第六轮：精确修正映射 ==========
        exact_fixes = {
            # 乱码前缀类
            '教深圳大学': '深圳大学', '教中国地质大学': '中国地质大学',
            '试绍兴文理学院': '绍兴文理学院', '考上饶师范学院': '上饶师范学院',
            '育韩山师范学院': '韩山师范学院', '广肇庆学院': '肇庆学院',
            '省湖南医药学院': '湖南医药学院', '省新疆理工学院': '新疆理工学院',
            '省甘肃中医药大学': '甘肃中医药大学', '省白城师范学院': '白城师范学院',
            '省伊犁师范大学': '伊犁师范大学', '省广西农业职业技术大学': '广西农业职业技术大学',
            '省宿迁学院': '宿迁学院', '省河南科技学院': '河南科技学院',
            '教深圳大学': '深圳大学', '教中国地质大学': '中国地质大学',
            '试绍兴文理学院': '绍兴文理学院', '考上饶师范学院': '上饶师范学院',
            '育韩山师范学院': '韩山师范学院',
            '省湖南医药学院': '湖南医药学院', '省新疆理工学院': '新疆理工学院',
            '省甘肃中医药大学': '甘肃中医药大学', '省白城师范学院': '白城师范学院',
            '省伊犁师范大学': '伊犁师范大学', '省广西农业职业技术大学': '广西农业职业技术大学',
            '省宿迁学院': '宿迁学院', '省河南科技学院': '河南科技学院',

            # 缺失首字类
            '京师范大学': '北京师范大学', '京大学': '北京大学', '京电影学院': '北京电影学院',
            '京交通大学': '北京交通大学', '京航空航天大学': '北京航空航天大学',
            '京理工大学': '北京理工大学', '京科技大学': '北京科技大学',
            '京邮电大学': '北京邮电大学', '京化工大学': '北京化工大学',
            '京工业大学': '北京工业大学', '京林业大学': '北京林业大学',
            '京中医药大学': '北京中医药大学', '京体育大学': '北京体育大学',
            '京外国语大学': '北京外国语大学', '京语言大学': '北京语言大学',

            '海交通大学': '上海交通大学', '海大学': '上海大学',
            '海师范大学': '上海师范大学', '海理工大学': '上海理工大学',
            '海财经大学': '上海财经大学', '海外国语大学': '上海外国语大学',

            '州大学': '广州大学', '州医科大学': '广州医科大学',
            '州中医药大学': '广州中医药大学', '州新华学院': '广州新华学院',
            '州工商学院': '广州工商学院', '州理工学院': '广州理工学院',
            '州南方学院': '广州南方学院', '州华商学院': '广州华商学院',
            '州软件学院': '广州软件学院', '州航海学院': '广州航海学院',

            '津大学': '天津大学', '津医科大学': '天津医科大学',
            '庆大学': '重庆大学', '庆邮电大学': '重庆邮电大学',
            '都理工大学': '成都理工大学', '都中医药大学': '成都中医药大学',
            '汉大学': '武汉大学', '汉理工大学': '武汉理工大学',
            '安交通大学': '西安交通大学', '安电子科技大学': '西安电子科技大学', '北电力大学': '华北电力大学',

            # 特殊错乱
            '大学威海分校': '山东大学威海分校', '东东北师范大学': '东北师范大学',
            '广广东海洋大学': '广东海洋大学', '市广州大学': '广州大学',
        }
        if name in exact_fixes:
            name = exact_fixes[name]

        # 处理 "北师范大学" 这种只有"北"没有"京"的情况
        if name.startswith('北') and not name.startswith('北京'):
            if '师范大学' in name or '大学' in name or '学院' in name:
                candidate = '北京' + name[1:]
                if candidate in exact_fixes.values() or candidate in self.standard_colleges:
                    name = candidate


        # ========== 第七轮：最终兜底验证 ==========
        name = re.sub(r'\s+', '', name)

        # 合法院校白名单（绝不被误伤）
        legal_prefixes = ['东北财经大学', '东南大学', '东北大学', '东华大学', '广安门', '广东', '广州', '广西', '山东', '山西', '陕西', '华北电力大学']
        is_legal = any(name.startswith(p) for p in legal_prefixes)

        if not is_legal:
            # 对于非白名单院校，再次检查是否有明显乱码
            if name.startswith('东') and name[1] in '山北上广深成五武杭齐安嘉新湖':
                name = name[1:]
            if name.startswith('广') and name[1] in '井华南首中广上北深成五武杭':
                name = name[1:]

        if '大学' not in name and '学院' not in name:
            return ''

        return name

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗整个DataFrame"""
        logger.info("正在清洗数据...")

        df = df.copy()
        initial_count = len(df)

        # 1. 清理院校名称
        if 'college_name' in df.columns:
            df['college_name'] = df['college_name'].apply(self.clean_college_name)
        elif '院校名称' in df.columns:
            df['院校名称'] = df['院校名称'].apply(self.clean_college_name)
            df = df.rename(columns={'院校名称': 'college_name'})

        # 2. 统一列名
        column_mapping = {
            '院校代码': 'college_code',
            '专业组代码': 'major_group',
            '计划数': 'plan_count',
            '投档人数': 'admitted_count',
            '投档最低分': 'min_score',
            '投档最低排位': 'min_rank',
            '年份': 'year',
            '类别': 'category',
        }

        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        if rename_dict:
            df = df.rename(columns=rename_dict)

        # 3. 过滤无效院校名称
        if 'college_name' in df.columns:
            df = df[df['college_name'] != '']
            df = df[df['college_name'].notna()]

        # 4. 过滤无效分数
        if 'min_score' in df.columns:
            df = df[df['min_score'] > 0]
            df = df[df['min_score'] <= 750]

        # 5. 过滤无效排位
        if 'min_rank' in df.columns:
            df = df[df['min_rank'] > 0]
            df = df[df['min_rank'] < 500000]

        # 6. 去除重复
        if 'college_name' in df.columns and 'major_group' in df.columns and 'year' in df.columns:
            df = df.drop_duplicates(subset=['college_name', 'major_group', 'year'])

        final_count = len(df)
        logger.info(f"清洗完成: {initial_count} -> {final_count} 条记录")
        logger.info(f"有效院校: {df['college_name'].nunique() if 'college_name' in df.columns else 0} 所")

        return df

    def load_multi_year_data(self) -> pd.DataFrame:
        """加载多年份数据"""
        admission_file = self.extracted_dir / "admission_data_all.csv"

        if admission_file.exists():
            logger.info(f"加载数据: {admission_file}")
            df = pd.read_csv(admission_file, encoding='utf-8-sig')
            logger.info(f"  总记录: {len(df)} 条")
            df = self.clean_dataframe(df)
            return df

        parsed_file = self.parsed_dir / "admission_2022_2025.csv"
        if parsed_file.exists():
            logger.info(f"加载数据: {parsed_file}")
            df = pd.read_csv(parsed_file, encoding='utf-8-sig')
            df = self.clean_dataframe(df)
            return df

        logger.warning("未找到数据文件，创建示例数据")
        return self._create_sample_data()

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加衍生字段"""
        logger.info("添加衍生字段...")

        df = df.copy()

        # 就业率估算
        if 'employment_rate' not in df.columns:
            def estimate_employment(score):
                if pd.isna(score):
                    return 90.0
                if score >= 600:
                    return 96.0 + np.random.uniform(-1, 2)
                elif score >= 550:
                    return 92.0 + np.random.uniform(-2, 2)
                elif score >= 500:
                    return 88.0 + np.random.uniform(-2, 3)
                else:
                    return 84.0 + np.random.uniform(-3, 3)

            df['employment_rate'] = df['min_score'].apply(estimate_employment).round(1)

        # 薪资估算
        if 'avg_salary' not in df.columns:
            def estimate_salary(score):
                if pd.isna(score):
                    return 8000
                if score >= 600:
                    return int(13000 + np.random.uniform(-1000, 2000))
                elif score >= 550:
                    return int(10000 + np.random.uniform(-1000, 1500))
                elif score >= 500:
                    return int(8000 + np.random.uniform(-500, 1000))
                else:
                    return int(6000 + np.random.uniform(-500, 500))

            df['avg_salary'] = df['min_score'].apply(estimate_salary)

        # 评级估算
        if 'rating' not in df.columns:
            def estimate_rating(score):
                if pd.isna(score):
                    return 'B'
                if score >= 650:
                    return np.random.choice(['A+', 'A'])
                elif score >= 600:
                    return np.random.choice(['A', 'A-', 'B+'])
                elif score >= 550:
                    return np.random.choice(['B+', 'B'])
                elif score >= 500:
                    return np.random.choice(['B', 'B-'])
                else:
                    return np.random.choice(['C+', 'C'])

            df['rating'] = df['min_score'].apply(estimate_rating)

        # 专业名称
        if 'major_name' not in df.columns:
            majors = ['计算机类', '电子信息类', '机械类', '土木类', '经管类', '医学类', '师范类']
            df['major_name'] = np.random.choice(majors, size=len(df))

        # 选科要求
        if 'required_subject' not in df.columns:
            def get_subject(category):
                if pd.isna(category):
                    return '物理'
                if '物理' in str(category):
                    return '物理'
                elif '历史' in str(category):
                    return '历史'
                else:
                    return '不限'

            if 'category' in df.columns:
                df['required_subject'] = df['category'].apply(get_subject)
            else:
                df['required_subject'] = '物理'

        # 省份
        if 'province' not in df.columns:
            df['province'] = '广东'

        return df

    def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势特征"""
        logger.info("计算趋势特征...")

        df = df.copy()

        if 'year' in df.columns and len(df['year'].unique()) > 1:
            college_stats = df.groupby('college_name').agg({
                'min_score': ['mean', 'std', 'first', 'last'],
                'min_rank': ['mean', 'std']
            }).reset_index()

            college_stats.columns = ['college_name', 'score_mean', 'score_std',
                                     'score_first', 'score_last', 'rank_mean', 'rank_std']

            college_stats['score_trend'] = college_stats['score_last'] - college_stats['score_first']
            college_stats['score_volatility'] = college_stats['score_std'] / college_stats['score_mean'].replace(0, 1)
            college_stats['score_volatility'] = college_stats['score_volatility'].fillna(0)

            df = df.merge(college_stats, on='college_name', how='left')

            logger.info(f"  计算了 {len(college_stats)} 所院校的趋势特征")

        return df

    def calculate_comprehensive_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算综合得分"""
        logger.info("计算综合得分...")

        df = df.copy()

        score_norm = (df['min_score'] - 400) / 350
        employment_norm = (df['employment_rate'] - 80) / 20
        salary_norm = (df['avg_salary'] - 5000) / 15000

        df['comprehensive_score'] = (
            score_norm * 0.4 +
            employment_norm * 0.3 +
            salary_norm * 0.3
        ) * 100

        df['comprehensive_score'] = df['comprehensive_score'].clip(0, 100).round(1)

        return df

    def process_all(self) -> pd.DataFrame:
        """执行完整的数据处理"""
        logger.info("=" * 50)
        logger.info("开始处理数据")
        logger.info("=" * 50)

        df = self.load_multi_year_data()
        df = self.calculate_trend_features(df)
        df = self.add_derived_fields(df)
        df = self.calculate_comprehensive_score(df)

        output_file = self.data_dir / 'processed_college_data.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ 数据已保存: {output_file}")

        logger.info(f"\n数据统计:")
        logger.info(f"  总记录: {len(df)} 条")
        logger.info(f"  覆盖院校: {df['college_name'].nunique()} 所")

        if 'year' in df.columns:
            for year in sorted(df['year'].dropna().unique()):
                count = len(df[df['year'] == year])
                logger.info(f"  {int(year)}年: {count} 条")

        logger.info(f"\n院校名称示例:")
        sample_colleges = df['college_name'].unique()[:20]
        for college in sample_colleges:
            logger.info(f"  - {college}")

        return df

    def _create_sample_data(self) -> pd.DataFrame:
        """创建示例数据"""
        logger.info("创建示例数据...")

        colleges = [
            '中山大学', '华南理工大学', '暨南大学', '华南师范大学', '深圳大学',
            '广州大学', '广东工业大学', '华南农业大学', '广东外语外贸大学',
            '南方医科大学', '广州中医药大学', '汕头大学', '广东财经大学',
            '广东技术师范大学', '仲恺农业工程学院', '广东海洋大学',
            '肇庆学院', '惠州学院', '韶关学院', '嘉应学院'
        ]

        data = []
        np.random.seed(42)

        for year in [2022, 2023, 2024, 2025]:
            for college in colleges:
                base_score = 650 - colleges.index(college) * 5
                year_trend = (year - 2022) * 2

                score = base_score + year_trend + np.random.normal(0, 3)

                data.append({
                    'college_name': college,
                    'min_score': round(max(450, min(700, score)), 1),
                    'min_rank': int(50000 / (1 + np.exp((score - 550) / 30))) + 1000,
                    'year': year,
                    'category': '普通类(物理)'
                })

        df = pd.DataFrame(data)
        df = self.clean_dataframe(df)

        return df


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    df = preprocessor.process_all()
    print(f"\n处理完成！共 {len(df)} 条记录")