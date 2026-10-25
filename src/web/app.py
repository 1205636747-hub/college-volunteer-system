"""
高考志愿推荐填报系统 - Web应用
"""

from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recommendation.recommender import CollegeRecommender
from src.recommendation.predictor import AdmissionPredictor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = 'college-recommendation-secret-key'

# 全局变量
recommender = None
predictor = None
college_data = None

# 数据目录
DATA_DIR = project_root / 'data'


def load_college_data():
    """加载院校数据"""
    global college_data

    # 尝试多个可能的数据文件
    possible_files = [
        DATA_DIR / 'processed_college_data.csv',
        DATA_DIR / 'parsed' / 'admission_2022_2025.csv',
        DATA_DIR / 'extracted_tables' / 'admission_data_all.csv',
    ]

    for file_path in possible_files:
        if file_path.exists():
            logger.info(f"加载数据: {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8-sig')

            # 确保必要的列存在
            if 'college_name' not in df.columns and '院校名称' in df.columns:
                df['college_name'] = df['院校名称']

            if 'college_name' not in df.columns:
                continue

            # 添加缺失的列
            if 'employment_rate' not in df.columns:
                df['employment_rate'] = 90.0
            if 'avg_salary' not in df.columns:
                df['avg_salary'] = 8000
            if 'rating' not in df.columns:
                df['rating'] = 'B'
            if 'major_name' not in df.columns:
                df['major_name'] = '待定专业'
            if 'required_subject' not in df.columns:
                df['required_subject'] = '不限'
            if 'min_score' not in df.columns and '投档最低分' in df.columns:
                df['min_score'] = df['投档最低分']
            if 'min_rank' not in df.columns and '投档最低排位' in df.columns:
                df['min_rank'] = df['投档最低排位']

            college_data = df
            logger.info(f"✓ 加载 {len(df)} 条记录")
            return True

    logger.warning("未找到数据文件，使用示例数据")
    college_data = create_sample_data()
    return True


def create_sample_data():
    """创建示例数据"""
    logger.info("创建示例数据...")

    colleges = [
        '中山大学', '华南理工大学', '暨南大学', '华南师范大学', '深圳大学',
        '广州大学', '广东工业大学', '华南农业大学', '广东外语外贸大学',
        '南方医科大学', '广州中医药大学', '汕头大学', '广东财经大学',
        '广东技术师范大学', '仲恺农业工程学院', '广东海洋大学'
    ]

    majors = ['计算机科学与技术', '软件工程', '电子信息工程', '工商管理', '会计学']

    data = []
    np.random.seed(42)

    for college in colleges:
        base_score = 650 - colleges.index(college) * 5

        for major in majors[:3]:
            data.append({
                'college_name': college,
                'major_name': major,
                'min_score': round(base_score + np.random.normal(0, 3), 1),
                'min_rank': int(50000 / (1 + np.exp((base_score - 550) / 30))) + 1000,
                'employment_rate': round(95 - colleges.index(college) * 0.3 + np.random.normal(0, 1), 1),
                'avg_salary': int(15000 - colleges.index(college) * 300 + np.random.normal(0, 500)),
                'rating': ['A+', 'A', 'A-', 'B+', 'B'][min(colleges.index(college) // 3, 4)],
                'province': '广东',
                'required_subject': '物理',
                'year': 2024
            })

    return pd.DataFrame(data)


def init_system():
    """初始化系统"""
    global recommender, predictor

    logger.info("正在初始化系统...")

    # 加载数据
    if not load_college_data():
        logger.error("数据加载失败")
        return False

    # 初始化推荐器
    recommender = CollegeRecommender()
    recommender.load_data(college_data)

    # 初始化预测器
    predictor = AdmissionPredictor()
    predictor.load_model()

    logger.info("系统初始化完成")
    return True


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/recommend')
def recommend_page():
    """推荐页面"""
    return render_template('recommend.html')


@app.route('/query')
def query_page():
    """数据查询页面"""
    return render_template('query.html')


@app.route('/analysis')
def analysis_page():
    """分析页面"""
    return render_template('analysis.html')


@app.route('/ai-assistant')
def ai_assistant_page():
    """AI助手页面"""
    return render_template('ai_assistant.html')


# ==================== 基础API路由 ====================

@app.route('/api/statistics')
def get_statistics():
    """获取统计信息"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    stats = {
        'total_colleges': college_data['college_name'].nunique() if 'college_name' in college_data.columns else 0,
        'total_majors': len(college_data),
        'avg_score_mean': round(college_data['min_score'].mean(), 1) if 'min_score' in college_data.columns else 0,
        'employment_rate_mean': round(college_data['employment_rate'].mean(), 1) if 'employment_rate' in college_data.columns else 90,
        'avg_salary_mean': round(college_data['avg_salary'].mean(), 0) if 'avg_salary' in college_data.columns else 8000
    }

    # 评分分布
    if 'rating' in college_data.columns:
        stats['rating_distribution'] = college_data['rating'].value_counts().to_dict()
    else:
        stats['rating_distribution'] = {'A': 10, 'B': 20, 'C': 5}

    # 分数分布
    if 'min_score' in college_data.columns:
        scores = college_data['min_score'].dropna()
        stats['score_distribution'] = {
            '500以下': len(scores[scores < 500]),
            '500-550': len(scores[(scores >= 500) & (scores < 550)]),
            '550-600': len(scores[(scores >= 550) & (scores < 600)]),
            '600-650': len(scores[(scores >= 600) & (scores < 650)]),
            '650以上': len(scores[scores >= 650])
        }
    else:
        stats['score_distribution'] = {'500以下': 5, '500-550': 10, '550-600': 15, '600-650': 10, '650以上': 5}

    return jsonify(stats)


@app.route('/api/subjects')
def get_subjects():
    """获取选科选项"""
    return jsonify({
        'subjects': {
            '物理': '物理',
            '化学': '化学',
            '生物': '生物',
            '历史': '历史',
            '地理': '地理',
            '政治': '政治'
        }
    })


@app.route('/api/provinces')
def get_provinces():
    """获取省份列表"""
    return jsonify({
        'provinces': ['广东', '北京', '上海', '江苏', '浙江', '湖北', '湖南', '四川', '陕西', '山东']
    })


@app.route('/api/colleges')
def get_colleges():
    """获取院校列表"""
    if college_data is not None and 'college_name' in college_data.columns:
        colleges = college_data['college_name'].unique().tolist()
        return jsonify({'colleges': colleges[:50]})
    return jsonify({'colleges': []})


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """获取推荐结果"""
    try:
        data = request.get_json()

        score = int(data.get('score', 0))
        rank = int(data.get('rank', 0))
        subjects = data.get('subjects', [])
        preference = data.get('preference', 'balanced')

        if score <= 0 or rank <= 0:
            return jsonify({'error': '请填写有效的分数和排位'}), 400

        logger.info(f"推荐请求: 分数={score}, 排位={rank}, 选科={subjects}")

        # 使用推荐引擎
        if recommender:
            result = recommender.recommend(score, rank, subjects, preference)
        else:
            result = {'rush': [], 'stable': [], 'safe': [], 'all': []}

        # 统计
        stats = {
            'rush_count': len(result.get('rush', [])),
            'stable_count': len(result.get('stable', [])),
            'safe_count': len(result.get('safe', []))
        }

        return jsonify({
            'success': True,
            'recommendations': result,
            'stats': stats
        })

    except Exception as e:
        logger.error(f"推荐失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== 数据查询API ====================

@app.route('/api/query/search', methods=['POST'])
def api_query_search():
    """综合数据查询接口"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    try:
        data = request.get_json()
        query_type = data.get('type', 'college')
        keyword = data.get('keyword', '').strip()
        score_min = data.get('score_min', None)
        score_max = data.get('score_max', None)
        year = data.get('year', None)
        page = data.get('page', 1)
        page_size = data.get('page_size', 20)

        df = college_data.copy()

        # 按查询类型筛选
        if query_type == 'college' and keyword:
            df = df[df['college_name'].str.contains(keyword, na=False, case=False)]
        elif query_type == 'major' and keyword:
            if 'major_name' in df.columns:
                df = df[df['major_name'].str.contains(keyword, na=False, case=False)]
            elif 'major_group' in df.columns:
                df = df[df['major_group'].astype(str).str.contains(keyword, na=False)]
        elif query_type == 'score':
            if score_min is not None:
                df = df[df['min_score'] >= float(score_min)]
            if score_max is not None:
                df = df[df['min_score'] <= float(score_max)]
        elif query_type == 'plan' and keyword:
            df = df[df['college_name'].str.contains(keyword, na=False, case=False)]

        # 按年份筛选
        if year and 'year' in df.columns:
            df = df[df['year'] == int(year)]

        # 分页
        total = len(df)
        total_pages = max(1, (total + page_size - 1) // page_size)
        start = (page - 1) * page_size
        end = start + page_size
        df_page = df.iloc[start:end]

        # 构建返回数据
        records = []
        for _, row in df_page.iterrows():
            records.append({
                'college_name': str(row.get('college_name', '')),
                'major_name': str(row.get('major_name', row.get('major_group', ''))),
                'min_score': float(row.get('min_score', 0)),
                'min_rank': int(row.get('min_rank', 0)) if pd.notna(row.get('min_rank')) else 0,
                'year': int(row.get('year', 0)) if pd.notna(row.get('year')) else 0,
                'category': str(row.get('category', '')),
                'plan_count': int(row.get('plan_count', 0)) if pd.notna(row.get('plan_count')) else 0,
                'admitted_count': int(row.get('admitted_count', 0)) if pd.notna(row.get('admitted_count')) else 0,
                'employment_rate': float(row.get('employment_rate', 0)),
                'avg_salary': int(row.get('avg_salary', 0)) if pd.notna(row.get('avg_salary')) else 0,
                'rating': str(row.get('rating', 'B')),
                'college_code': str(row.get('college_code', '')),
            })

        return jsonify({
            'success': True,
            'records': records,
            'total': total,
            'page': page,
            'total_pages': total_pages,
            'page_size': page_size
        })

    except Exception as e:
        logger.error(f"查询失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query/college-detail', methods=['POST'])
def api_college_detail():
    """获取院校详细信息"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    try:
        data = request.get_json()
        college_name = data.get('college_name', '').strip()

        if not college_name:
            return jsonify({'error': '院校名称不能为空'}), 400

        df = college_data[college_data['college_name'] == college_name].copy()

        if len(df) == 0:
            return jsonify({'error': '未找到该院校信息'}), 404

        info = {
            'college_name': college_name,
            'college_code': str(df['college_code'].iloc[0]) if 'college_code' in df.columns else '',
            'rating': str(df['rating'].iloc[0]) if 'rating' in df.columns else 'B',
            'employment_rate': float(df['employment_rate'].mean()) if 'employment_rate' in df.columns else 0,
            'avg_salary': int(df['avg_salary'].mean()) if 'avg_salary' in df.columns else 0,
        }

        if 'major_name' in df.columns:
            info['majors'] = df['major_name'].unique().tolist()[:10]
        elif 'major_group' in df.columns:
            info['majors'] = df['major_group'].unique().tolist()[:10]
        else:
            info['majors'] = []

        # 历年招生计划统计
        if 'year' in df.columns:
            yearly_plans = df.groupby('year')['plan_count'].sum().to_dict() if 'plan_count' in df.columns else {}
            info['yearly_plans'] = {str(int(k)): int(v) for k, v in yearly_plans.items()}

        return jsonify({'success': True, 'info': info})

    except Exception as e:
        logger.error(f"获取院校详情失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query/years')
def api_query_years():
    """获取可用年份列表"""
    if college_data is not None and 'year' in college_data.columns:
        years = sorted(college_data['year'].dropna().unique().astype(int).tolist())
        return jsonify({'years': years})
    return jsonify({'years': [2022, 2023, 2024, 2025]})


# ==================== 数据分析API ====================

@app.route('/api/analysis/score-trend')
def api_score_trend():
    """获取院校分数趋势数据"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    top_colleges = college_data['college_name'].value_counts().head(15).index.tolist()

    result = []
    for college in top_colleges:
        college_df = college_data[college_data['college_name'] == college]
        if 'year' in college_df.columns and 'min_score' in college_df.columns:
            yearly_data = college_df.groupby('year')['min_score'].mean().round(1).sort_index()
            if len(yearly_data) >= 2:
                result.append({
                    'name': college,
                    'years': yearly_data.index.astype(int).tolist(),
                    'scores': yearly_data.values.tolist()
                })

    return jsonify({'colleges': result[:10]})


@app.route('/api/analysis/rank-trend')
def api_rank_trend():
    """获取院校位次走势数据"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    top_colleges = college_data['college_name'].value_counts().head(15).index.tolist()

    result = []
    for college in top_colleges:
        college_df = college_data[college_data['college_name'] == college]
        if 'year' in college_df.columns and 'min_rank' in college_df.columns:
            yearly_data = college_df.groupby('year')['min_rank'].mean().round(0).sort_index()
            if len(yearly_data) >= 2:
                result.append({
                    'name': college,
                    'years': yearly_data.index.astype(int).tolist(),
                    'ranks': yearly_data.values.astype(int).tolist()
                })

    return jsonify({'colleges': result[:10]})


@app.route('/api/analysis/plan-distribution')
def api_plan_distribution():
    """获取招生计划分布数据"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    if 'min_score' in college_data.columns:
        df = college_data.copy()
        bins = [0, 450, 500, 550, 600, 650, 750]
        labels = ['450以下', '450-500', '500-550', '550-600', '600-650', '650以上']
        df['score_range'] = pd.cut(df['min_score'], bins=bins, labels=labels)

        if 'plan_count' in df.columns:
            plan_stats = df.groupby('score_range')['plan_count'].sum().fillna(0).astype(int)
        else:
            plan_stats = df.groupby('score_range').size()

        result = {
            'categories': labels,
            'values': [int(plan_stats.get(label, 0)) for label in labels]
        }

        if 'year' in df.columns:
            year_stats = {}
            for year in sorted(df['year'].dropna().unique()):
                year_df = df[df['year'] == year]
                if 'plan_count' in year_df.columns:
                    year_stats[str(int(year))] = int(year_df['plan_count'].sum())
                else:
                    year_stats[str(int(year))] = len(year_df)
            result['year_stats'] = year_stats

        return jsonify(result)

    return jsonify({'error': '数据不足'}), 404


@app.route('/api/analysis/major-heat')
def api_major_heat():
    """获取专业热度数据"""
    if college_data is None:
        return jsonify({'error': '数据未加载'}), 500

    if 'major_name' in college_data.columns:
        major_counts = college_data['major_name'].value_counts().head(20)
        result = {
            'majors': major_counts.index.tolist(),
            'counts': major_counts.values.tolist()
        }
        return jsonify(result)

    if 'major_group' in college_data.columns:
        group_counts = college_data['major_group'].value_counts().head(20)
        result = {
            'majors': [f'专业组{str(g)}' for g in group_counts.index],
            'counts': group_counts.values.tolist()
        }
        return jsonify(result)

    return jsonify({'error': '数据不足'}), 404


@app.route('/api/analysis/year-comparison')
def api_year_comparison():
    """获取各年份数据对比"""
    if college_data is None or 'year' not in college_data.columns:
        return jsonify({'error': '数据不足'}), 404

    df = college_data.copy()

    result = {
        'years': [],
        'avg_scores': [],
        'avg_ranks': [],
        'college_counts': [],
        'record_counts': []
    }

    for year in sorted(df['year'].dropna().unique()):
        year_df = df[df['year'] == year]
        result['years'].append(str(int(year)))
        result['avg_scores'].append(round(year_df['min_score'].mean(), 1))
        result['avg_ranks'].append(int(year_df['min_rank'].mean()))
        result['college_counts'].append(year_df['college_name'].nunique())
        result['record_counts'].append(len(year_df))

    return jsonify(result)


# ==================== AI助手路由 ====================

@app.route('/api/ai/status')
def ai_status():
    """AI服务状态"""
    try:
        from src.ai.doubao_client import get_doubao_client
        client = get_doubao_client()
        return jsonify(client.get_status())
    except Exception as e:
        logger.warning(f"AI状态获取失败: {e}")
        return jsonify({
            'available': True,
            'mode': 'offline',
            'model': 'offline-simulator',
            'free_quota_remaining': '无限（离线模式）'
        })


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """AI对话"""
    try:
        from src.ai.doubao_client import get_doubao_client

        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})
        stream = data.get('stream', False)

        if not message:
            return jsonify({'error': '消息不能为空'}), 400

        client = get_doubao_client()

        if stream:
            def generate():
                for chunk in client.chat_stream(message, context):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate(), mimetype='text/event-stream')
        else:
            response = client.chat(message, context)
            return jsonify({'success': True, 'response': response})

    except ImportError as e:
        logger.warning(f"AI模块未安装: {e}")
        # 降级响应
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})

        score = context.get('score')
        rank = context.get('rank')

        if score and rank:
            if score >= 600:
                response = "🚀 冲：中山大学、华南理工大学\n⚖️ 稳：深圳大学、华南师范大学\n🛡️ 保：广州大学、广东工业大学"
            elif score >= 550:
                response = "🚀 冲：深圳大学、华南师范大学\n⚖️ 稳：广州大学、广东工业大学\n🛡️ 保：广东财经大学、东莞理工学院"
            elif score >= 500:
                response = "🚀 冲：广州大学、广东工业大学\n⚖️ 稳：广东技术师范大学、仲恺农业工程学院\n🛡️ 保：肇庆学院、惠州学院"
            else:
                response = "🚀 冲：广东技术师范大学、五邑大学\n⚖️ 稳：肇庆学院、韶关学院\n🛡️ 保：民办本科院校"
        else:
            response = "👋 您好！我是AI志愿助手。请告诉我您的分数和排位，我会为您推荐合适的院校！"

        return jsonify({'success': True, 'response': response})
    except Exception as e:
        logger.error(f"AI对话失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/clear', methods=['POST'])
def ai_clear_history():
    """清空对话历史"""
    try:
        from src.ai.doubao_client import get_doubao_client
        client = get_doubao_client()
        client.clear_history()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"清空历史失败: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': '服务器错误'}), 500


# 初始化系统
init_system()


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)