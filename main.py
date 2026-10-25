"""
高考志愿推荐填报系统 - 系统入口
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("🎓 高考志愿推荐填报系统")
    print("=" * 60)

    print(f"\n项目根目录: {project_root.absolute()}")
    print(f"\n访问地址:")
    print(f"  首页: http://127.0.0.1:5000")
    print(f"  推荐: http://127.0.0.1:5000/recommend")
    print(f"  分析: http://127.0.0.1:5000/analysis")
    print(f"  AI助手: http://127.0.0.1:5000/ai-assistant")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 60)

    try:
        from src.web.app import app
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        logger.error(f"启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()