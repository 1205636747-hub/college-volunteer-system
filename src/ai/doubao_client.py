"""
豆包AI客户端 - 支持真实API调用与内容过滤
"""

import json
import logging
import re
import requests
from typing import Dict, List, Optional, Generator
from pathlib import Path
import sys

# 添加项目路径以读取配置
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import DOUBAO_CONFIG, OFFLINE_MODE_ENABLED
except ImportError:
    DOUBAO_CONFIG = {}
    OFFLINE_MODE_ENABLED = True

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DoubaoClient:
    """豆包AI客户端 - 支持API调用与智能内容过滤"""

    def __init__(self, api_key: str = None, model: str = None):
        # 读取配置
        self.api_key = api_key or DOUBAO_CONFIG.get('api_key', '')
        self.base_url = DOUBAO_CONFIG.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3')
        self.model = model or DOUBAO_CONFIG.get('model', 'doubao-lite-32k')
        self.temperature = DOUBAO_CONFIG.get('temperature', 0.7)
        self.max_tokens = DOUBAO_CONFIG.get('max_tokens', 2048)
        self.stream_enabled = DOUBAO_CONFIG.get('stream', True)

        # 离线模式判断
        self.offline_mode = OFFLINE_MODE_ENABLED or not self.api_key
        self.conversation_history: List[Dict] = []

        if self.offline_mode:
            logger.info("🔵 豆包客户端运行在离线模拟模式")
        else:
            logger.info(f"🟢 豆包客户端运行在API模式，模型: {self.model}")

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return not self.offline_mode

    def get_status(self) -> Dict:
        """获取服务状态"""
        return {
            'available': not self.offline_mode,
            'mode': 'offline' if self.offline_mode else 'api',
            'model': self.model if not self.offline_mode else 'offline-simulator',
            'free_quota_remaining': '无限（离线模式）' if self.offline_mode else '50万Tokens免费额度'
        }

    # ==================== 内容过滤与提取 ====================
    def _clean_response(self, text: str) -> str:
        """
        清理API返回的冗余内容，只保留有用的志愿相关信息
        """
        if not text:
            return ""

        # 1. 去除常见的无关前缀
        prefixes_to_remove = [
            "根据您的成绩分析",
            "以下是推荐结果",
            "您好！",
            "当然可以",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip('：:')

        # 2. 去除过长的空格和换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # 3. 如果包含明显的推荐内容标记，提取核心部分
        # 提取"冲一冲"、"稳一稳"、"保一保"所在段落
        important_sections = []
        lines = text.split('\n')
        capture = False
        for line in lines:
            if any(keyword in line for keyword in ['冲一冲', '稳一稳', '保一保', '录取概率', '推荐院校']):
                capture = True
            if capture:
                important_sections.append(line)
            # 遇到空行且已捕获内容后，若下一行不是推荐相关则停止
            if capture and line.strip() == '' and len(important_sections) > 2:
                # 检查下一行
                continue

        if important_sections:
            return '\n'.join(important_sections).strip()

        # 4. 如果没有明显结构，返回原文但限制长度
        if len(text) > 1500:
            # 尝试按句号分割取前几句
            sentences = re.split(r'[。！？]', text)
            short_text = '。'.join(sentences[:8]) + '。'
            return short_text

        return text.strip()

    def _build_system_prompt(self) -> str:
        """构建系统提示词，引导模型输出简洁内容"""
        return """你是一个专业的高考志愿填报助手，专门为广东省考生提供服务。

你的回答必须简洁、专业、直接，只输出与志愿填报相关的核心内容。

规则：
1. 如果用户询问分数对应的院校，直接给出"冲、稳、保"三个层次的具体院校名称，不要多余描述。
2. 如果用户询问专业前景，用2-3句话概括核心优势和就业方向。
3. 如果用户询问院校信息，用3-4个要点介绍院校层次、优势专业、录取大致排位。
4. 不要输出问候语、客套话、冗长背景介绍。
5. 使用emoji适度点缀，但不要过多。

输出格式示例：
🚀 冲：中山大学(部分专业)、华南理工大学
⚖️ 稳：深圳大学、华南师范大学
🛡️ 保：广州大学、广东工业大学
"""

    # ==================== API调用 ====================
    def _call_api(self, messages: List[Dict]) -> str:
        """调用豆包API（非流式）"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"API调用成功，返回长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise

    def _call_api_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """调用豆包API（流式）"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"API流式调用失败: {e}")
            raise

    # ==================== 对话接口 ====================
    def chat(self, user_message: str, context: Dict = None) -> str:
        """发送消息并获取回复"""
        # 保存用户消息
        self.conversation_history.append({"role": "user", "content": user_message})

        # 离线模式
        if self.offline_mode:
            response = self._offline_response(user_message, context)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        # 构建消息
        messages = [{"role": "system", "content": self._build_system_prompt()}]

        # 添加上下文
        if context:
            context_str = self._format_context(context)
            if context_str:
                messages.append({"role": "system", "content": context_str})

        # 添加历史（最近6轮）
        for msg in self.conversation_history[-12:]:
            messages.append(msg)

        try:
            raw_response = self._call_api(messages)
            cleaned = self._clean_response(raw_response)
            self.conversation_history.append({"role": "assistant", "content": cleaned})
            return cleaned
        except Exception as e:
            logger.error(f"API调用异常，降级到离线模式: {e}")
            response = self._offline_response(user_message, context)
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

    def chat_stream(self, user_message: str, context: Dict = None) -> Generator[str, None, None]:
        """流式对话"""
        self.conversation_history.append({"role": "user", "content": user_message})

        if self.offline_mode:
            response = self._offline_response(user_message, context)
            self.conversation_history.append({"role": "assistant", "content": response})
            yield response
            return

        messages = [{"role": "system", "content": self._build_system_prompt()}]
        if context:
            context_str = self._format_context(context)
            if context_str:
                messages.append({"role": "system", "content": context_str})
        for msg in self.conversation_history[-12:]:
            messages.append(msg)

        full_response = ""
        try:
            for chunk in self._call_api_stream(messages):
                full_response += chunk
                yield chunk
            cleaned = self._clean_response(full_response)
            self.conversation_history.append({"role": "assistant", "content": cleaned})
        except Exception as e:
            logger.error(f"API流式调用异常: {e}")
            fallback = self._offline_response(user_message, context)
            self.conversation_history.append({"role": "assistant", "content": fallback})
            yield fallback

    def _format_context(self, context: Dict) -> str:
        """格式化上下文信息"""
        if not context:
            return ""
        parts = []
        if context.get('score'):
            parts.append(f"考生分数：{context['score']}分")
        if context.get('rank'):
            parts.append(f"全省排位：第{context['rank']}名")
        if context.get('province'):
            parts.append(f"省份：{context['province']}")
        if context.get('subjects'):
            parts.append(f"选科：{', '.join(context['subjects'])}")
        if parts:
            return "当前考生信息：" + "，".join(parts)
        return ""

    # ==================== 离线模拟（备胎） ====================
    def _offline_response(self, user_message: str, context: Dict = None) -> str:
        """离线模式下的规则响应"""
        message_lower = user_message.lower()
        score = context.get('score') if context else None
        rank = context.get('rank') if context else None

        if '冲稳保' in user_message or '策略' in user_message:
            return self._explain_strategy()
        if '计算机' in user_message:
            return self._explain_major('计算机')
        if '选科' in user_message:
            return self._explain_subject_selection()
        if '大学' in user_message and '怎么样' in user_message:
            return self._introduce_college(user_message)

        if score and rank:
            return self._generate_recommendation(score, rank)

        return "请提供您的分数和排位，或直接问我具体问题（如'计算机专业怎么样'、'冲稳保策略'）。"

    def _generate_recommendation(self, score: int, rank: int) -> str:
        """生成简洁推荐"""
        if score >= 650:
            return f"🚀 冲：清华、北大、复旦、浙大\n⚖️ 稳：中山大学、华南理工、武大\n🛡️ 保：暨南大学、深圳大学、南方医科"
        elif score >= 600:
            return f"🚀 冲：中山大学、华南理工、暨大热门专业\n⚖️ 稳：深圳大学、华师、广外\n🛡️ 保：广州大学、广工、华农"
        elif score >= 550:
            return f"🚀 冲：深圳大学、华师、广州医科\n⚖️ 稳：广州大学、广工、华农\n🛡️ 保：广技师、仲恺、五邑大学"
        elif score >= 500:
            return f"🚀 冲：广技师、广东金融、五邑\n⚖️ 稳：肇庆学院、韶关学院、嘉应\n🛡️ 保：民办本科或应用型院校"
        else:
            return f"🚀 冲：民办本科优势专业\n⚖️ 稳：专科热门专业\n🛡️ 保：确保录取的院校"

    def _explain_strategy(self) -> str:
        return "🎯 冲(30%-60%)：往年排位略高的院校；稳(60%-80%)：排位相近；保(80%+)：排位低于自己的院校。比例建议10:20:15。"

    def _explain_major(self, major: str) -> str:
        if '计算机' in major:
            return "💻 计算机类：就业好、薪资高。优势院校：中山、华工、深大、广工。适合数学逻辑强的考生。"
        return "📖 请告诉我具体专业名称。"

    def _explain_subject_selection(self) -> str:
        return "🔬 物化生组合专业覆盖率最高(96%+)。工科/医学必选物理+化学；法学/经管不限。"

    def _introduce_college(self, query: str) -> str:
        if '中山大学' in query:
            return "🏫 中山大学：985/211，广州/珠海/深圳。王牌：临床医学、工商管理。物理类约635分/8000名。"
        if '华南理工' in query:
            return "🏫 华南理工大学：985/211，广州。建筑老八校，轻工全国第一。物理类约620分/15000名。"
        if '深圳大学' in query:
            return "🏫 深圳大学：双非顶尖，深圳。计算机/电子强势，就业好。物理类约590分/30000名。"
        return "请指定院校名称（如'中山大学怎么样'）。"

    def clear_history(self):
        self.conversation_history = []


_doubao_client: Optional[DoubaoClient] = None


def get_doubao_client() -> DoubaoClient:
    global _doubao_client
    if _doubao_client is None:
        _doubao_client = DoubaoClient()
    return _doubao_client