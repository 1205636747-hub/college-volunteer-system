"""
通用PDF表格提取工具 - 支持连续数字格式（2023年）
"""

import pdfplumber
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class UniversalPDFExtractor:
    """通用PDF提取器"""

    def __init__(self, pdf_dir: str = "data/raw_pdfs", output_dir: str = "data/extracted_tables"):
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent
        self.pdf_dir = self.project_root / pdf_dir
        self.output_dir = self.project_root / output_dir

        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PDF目录: {self.pdf_dir.absolute()}")
        logger.info(f"输出目录: {self.output_dir.absolute()}")

    def extract_score_distribution(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """
        提取分数段统计数据
        支持连续数字格式：58023232323579225225...
        """
        logger.info(f"提取分数段: {pdf_path.name}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""

                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

                if not all_text:
                    logger.warning(f"  未提取到文本")
                    return None

                # 提取所有数字（包括连续数字）
                scores_data = self._parse_continuous_numbers(all_text)

                if not scores_data:
                    logger.warning(f"  无法解析数字序列")
                    return None

                df = pd.DataFrame(scores_data)
                df = df.sort_values('分数', ascending=False).reset_index(drop=True)

                logger.info(f"  ✓ 提取 {len(df)} 个分数段 (分数: {df['分数'].max()} - {df['分数'].min()})")

                return df

        except Exception as e:
            logger.error(f"  提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_continuous_numbers(self, text: str) -> List[Dict]:
        """
        解析连续数字文本
        根据数字的合理范围来切分
        """
        # 移除所有非数字字符
        numbers_str = re.sub(r'[^\d]', '', text)

        if not numbers_str:
            return []

        logger.info(f"  提取到连续数字串，长度: {len(numbers_str)}")

        # 方法1：智能切分
        results = self._smart_split_numbers(numbers_str)

        # 方法2：如果方法1结果太少，尝试固定长度切分
        if len(results) < 10:
            results = self._fixed_length_split(numbers_str)

        return results

    def _smart_split_numbers(self, numbers_str: str) -> List[Dict]:
        """
        智能切分：根据数字的合理范围
        """
        results = []
        i = 0
        n = len(numbers_str)

        while i < n:
            # 尝试提取分数
            score, score_len = self._extract_score(numbers_str, i)

            if score is None:
                i += 1
                continue

            # 尝试提取后续的4个数字
            numbers = []
            pos = i + score_len

            for _ in range(4):
                num, num_len = self._extract_number(numbers_str, pos)
                if num is not None:
                    numbers.append(num)
                    pos += num_len
                else:
                    break

            if len(numbers) >= 4:
                # 验证累计人数是否合理
                if numbers[1] >= numbers[0] and numbers[3] >= numbers[2]:
                    results.append({
                        '分数': score,
                        '本科人数': numbers[0],
                        '本科累计人数': numbers[1],
                        '专科人数': numbers[2],
                        '专科累计人数': numbers[3]
                    })
                    i = pos
                else:
                    i += 1
            elif len(numbers) >= 2:
                # 可能只有本科数据
                results.append({
                    '分数': score,
                    '本科人数': numbers[0] if len(numbers) > 0 else 0,
                    '本科累计人数': numbers[1] if len(numbers) > 1 else 0,
                    '专科人数': 0,
                    '专科累计人数': 0
                })
                i = pos
            else:
                i += 1

        return results

    def _extract_score(self, text: str, start: int) -> Tuple[Optional[int], int]:
        """
        从指定位置提取分数
        分数特征：3位数，范围200-750 或 300-600
        """
        if start + 3 > len(text):
            return None, 0

        # 尝试3位数
        score_str = text[start:start+3]
        if score_str.isdigit():
            score = int(score_str)
            # 普通类：200-750，艺术类：300-600
            if 200 <= score <= 750:
                return score, 3
            if 300 <= score <= 600:
                return score, 3

        return None, 0

    def _extract_number(self, text: str, start: int) -> Tuple[Optional[int], int]:
        """
        从指定位置提取一个数字（可能是1-6位）
        """
        if start >= len(text):
            return None, 0

        # 尝试不同长度
        for length in [6, 5, 4, 3, 2, 1]:
            if start + length > len(text):
                continue

            num_str = text[start:start+length]
            if num_str.isdigit():
                num = int(num_str)
                # 累计人数通常较大，可能达到几十万
                if num > 0:
                    return num, length

        return None, 0

    def _fixed_length_split(self, numbers_str: str) -> List[Dict]:
        """
        固定长度切分：每5个数字一组
        通过分析数字长度模式来切分
        """
        # 尝试找出数字的分界点
        # 分数通常是3位数，累计人数可能是2-6位数

        results = []
        i = 0
        n = len(numbers_str)

        while i + 3 <= n:
            score_str = numbers_str[i:i+3]
            if not score_str.isdigit():
                i += 1
                continue

            score = int(score_str)
            if not (200 <= score <= 750):
                i += 1
                continue

            # 找到了有效分数，尝试找后续数字
            pos = i + 3
            numbers = []

            # 收集直到遇到下一个可能的分数
            while pos < n and len(numbers) < 4:
                # 检查是否遇到下一个分数
                if pos + 3 <= n:
                    next_score_str = numbers_str[pos:pos+3]
                    if next_score_str.isdigit():
                        next_score = int(next_score_str)
                        if 200 <= next_score <= 750:
                            break

                # 继续收集当前数字
                pos += 1

            # 从i+3到pos之间的数字
            middle = numbers_str[i+3:pos]
            if middle:
                # 尝试解析中间的数字
                nums = self._parse_middle_numbers(middle)
                if len(nums) >= 2:
                    results.append({
                        '分数': score,
                        '本科人数': nums[0] if len(nums) > 0 else 0,
                        '本科累计人数': nums[1] if len(nums) > 1 else 0,
                        '专科人数': nums[2] if len(nums) > 2 else 0,
                        '专科累计人数': nums[3] if len(nums) > 3 else 0
                    })

            i = pos

        return results

    def _parse_middle_numbers(self, middle: str) -> List[int]:
        """解析中间的数字部分"""
        numbers = []

        # 根据常见长度模式切分
        # 人数通常2-3位，累计人数可能3-6位

        i = 0
        n = len(middle)

        while i < n:
            # 尝试提取一个合理的数字
            found = False
            for length in [2, 3, 4, 5]:
                if i + length > n:
                    continue
                num_str = middle[i:i+length]
                if num_str.isdigit():
                    numbers.append(int(num_str))
                    i += length
                    found = True
                    break

            if not found:
                i += 1

        return numbers

    def extract_admission_data(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """提取投档情况数据"""
        logger.info(f"提取投档数据: {pdf_path.name}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_records = []

                for page in pdf.pages:
                    tables = page.extract_tables()

                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        records = self._parse_admission_table(table)
                        all_records.extend(records)

                if not all_records:
                    # 尝试从文本提取
                    text = ""
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"
                    all_records = self._parse_admission_text(text)

                if not all_records:
                    logger.warning(f"  未提取到投档数据")
                    return None

                df = pd.DataFrame(all_records)
                df = df[df['投档最低分'] > 0]

                logger.info(f"  ✓ 提取 {len(df)} 条投档记录")
                return df

        except Exception as e:
            logger.error(f"  提取失败: {e}")
            return None

    def _parse_admission_table(self, table: List[List]) -> List[Dict]:
        """解析投档情况表格"""
        records = []

        clean_table = []
        for row in table:
            if row and any(cell for cell in row if cell):
                clean_row = [self._clean_cell(cell) for cell in row]
                clean_table.append(clean_row)

        for row in clean_table:
            if not row or not row[0]:
                continue

            college_code = row[0]
            if not college_code.isdigit() or len(college_code) != 5:
                continue

            record = self._extract_admission_row(row)
            if record:
                records.append(record)

        return records

    def _parse_admission_text(self, text: str) -> List[Dict]:
        """从文本解析投档数据"""
        records = []

        # 匹配模式：5位院校代码 + 院校名称 + 专业组代码 + 数字...
        lines = text.split('\n')

        for line in lines:
            # 查找5位数字开头的行
            match = re.match(r'^(\d{5})\s+', line)
            if match:
                parts = line.split()
                if len(parts) >= 7:
                    record = {
                        '院校代码': parts[0],
                        '院校名称': parts[1] if len(parts) > 1 else '',
                        '专业组代码': parts[2] if len(parts) > 2 else '',
                        '计划数': self._parse_int(parts[3]) if len(parts) > 3 else 0,
                        '投档人数': self._parse_int(parts[4]) if len(parts) > 4 else 0,
                        '投档最低分': self._parse_int(parts[5]) if len(parts) > 5 else 0,
                        '投档最低排位': self._parse_int(parts[6]) if len(parts) > 6 else 0
                    }
                    records.append(record)

        return records

    def _extract_admission_row(self, row: List[str]) -> Optional[Dict]:
        """提取单行投档数据"""
        if len(row) < 7:
            return None

        return {
            '院校代码': row[0] if len(row) > 0 else '',
            '院校名称': row[1] if len(row) > 1 else '',
            '专业组代码': row[2] if len(row) > 2 else '',
            '计划数': self._parse_int(row[3]) if len(row) > 3 else 0,
            '投档人数': self._parse_int(row[4]) if len(row) > 4 else 0,
            '投档最低分': self._parse_int(row[5]) if len(row) > 5 else 0,
            '投档最低排位': self._parse_int(row[6]) if len(row) > 6 else 0
        }

    def _parse_int(self, value) -> int:
        """解析整数"""
        if value == '--' or value == '' or value is None:
            return 0
        try:
            return int(str(value).replace(',', ''))
        except:
            return 0

    def _clean_cell(self, cell) -> str:
        if cell is None:
            return ""
        return str(cell).strip()

    def extract_year_and_category(self, filename: str) -> Tuple[Optional[int], str]:
        """从文件名提取年份和类别"""
        year_match = re.search(r'(\d{4})年', filename)
        year = int(year_match.group(1)) if year_match else None

        if '普通类' in filename:
            if '历史' in filename:
                category = '普通类(历史)'
            elif '物理' in filename:
                category = '普通类(物理)'
            else:
                category = '普通类'
        elif '体育' in filename:
            category = '体育类'
        elif '美术' in filename or '设计' in filename:
            category = '美术类'
        elif '音乐' in filename:
            category = '音乐类'
        elif '舞蹈' in filename:
            category = '舞蹈类'
        elif '书法' in filename:
            category = '书法类'
        elif '播音' in filename:
            category = '播音类'
        elif '广播电视' in filename or '编导' in filename:
            category = '编导类'
        elif '表演' in filename or '表(导)演' in filename:
            category = '表演类'
        else:
            category = '其他'

        return year, category

    def process_all_pdfs(self) -> Dict[str, pd.DataFrame]:
        """处理所有PDF文件"""
        logger.info("=" * 70)
        logger.info("开始批量处理PDF文件")
        logger.info("=" * 70)

        all_admission = []
        all_scores = []

        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        # 优先处理2023年的分数段文件（问题文件）
        pdf_files.sort(key=lambda x: (
            0 if '2023' in x.name and '分数段' in x.name else 1,
            x.name
        ))

        logger.info(f"找到 {len(pdf_files)} 个PDF文件\n")

        for pdf_file in pdf_files:
            filename = pdf_file.name
            year, category = self.extract_year_and_category(filename)

            if '投档' in filename:
                df = self.extract_admission_data(pdf_file)
                if df is not None and not df.empty:
                    df['年份'] = year
                    df['类别'] = category
                    all_admission.append(df)

            elif '分数段' in filename or '统计表' in filename:
                logger.info(f"\n处理: {filename}")
                df = self.extract_score_distribution(pdf_file)
                if df is not None and not df.empty:
                    df['年份'] = year
                    df['类别'] = category
                    all_scores.append(df)
                    logger.info(f"  成功提取 {len(df)} 条记录")
                else:
                    logger.warning(f"  提取失败")

        result = {}

        if all_admission:
            admission_df = pd.concat(all_admission, ignore_index=True)
            csv_path = self.output_dir / "admission_data_all.csv"
            admission_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 投档数据已保存: {csv_path}")
            result['admission'] = admission_df

        if all_scores:
            scores_df = pd.concat(all_scores, ignore_index=True)
            csv_path = self.output_dir / "score_distribution_all.csv"
            scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 分数段数据已保存: {csv_path}")
            result['score'] = scores_df

        return result


def main():
    print("\n" + "=" * 70)
    print("  📊 通用PDF表格提取工具（支持连续数字）")
    print("=" * 70)

    extractor = UniversalPDFExtractor(
        pdf_dir="data/raw_pdfs",
        output_dir="data/extracted_tables"
    )

    result = extractor.process_all_pdfs()

    if result.get('score') is not None:
        print(f"\n✅ 分数段数据提取成功！共 {len(result['score'])} 条记录")
    if result.get('admission') is not None:
        print(f"✅ 投档数据提取成功！共 {len(result['admission'])} 条记录")


if __name__ == '__main__':
    main()