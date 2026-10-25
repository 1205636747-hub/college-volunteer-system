"""
通用PDF表格提取工具 - 支持广东省教育考试院所有PDF格式
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
    """通用PDF提取器 - 支持所有年份所有类别的PDF格式"""

    def __init__(self, pdf_dir: str = "data/raw_pdfs", output_dir: str = "data/extracted_tables"):
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent
        self.pdf_dir = self.project_root / pdf_dir
        self.output_dir = self.project_root / output_dir

        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PDF目录: {self.pdf_dir.absolute()}")
        logger.info(f"输出目录: {self.output_dir.absolute()}")

    # ==================== 分数段统计表提取 ====================

    def extract_score_distribution(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """
        提取分数段统计数据
        支持紧凑格式：58023232323579225225578...
        每5个数字一组：分数、本科人数、本科累计、专科人数、专科累计
        """
        logger.info(f"提取分数段: {pdf_path.name}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_numbers = []

                for page in pdf.pages:
                    # 提取页面文本
                    text = page.extract_text()
                    if not text:
                        continue

                    # 提取所有数字
                    numbers = re.findall(r'\b(\d+)\b', text)
                    all_numbers.extend([int(n) for n in numbers])

                if not all_numbers:
                    logger.warning(f"  未提取到数字")
                    return None

                # 过滤掉非分数段的数字（如表头中的年份等）
                # 分数段数据的特征：连续的3位数字
                scores_data = self._parse_score_sequence(all_numbers)

                if not scores_data:
                    logger.warning(f"  无法解析分数段序列")
                    return None

                df = pd.DataFrame(scores_data)

                # 按分数降序排列
                df = df.sort_values('分数', ascending=False).reset_index(drop=True)

                # 验证累计人数
                df['本科累计_验证'] = df['本科人数'].cumsum()
                df['专科累计_验证'] = df['专科人数'].cumsum()

                logger.info(f"  ✓ 提取 {len(df)} 个分数段 (分数: {df['分数'].max()} - {df['分数'].min()})")

                return df

        except Exception as e:
            logger.error(f"  提取失败: {e}")
            return None

    def _parse_score_sequence(self, numbers: List[int]) -> List[Dict]:
        """从数字序列中解析分数段数据"""
        results = []

        # 分数段的特征：
        # 1. 分数通常在200-750之间（普通类）或300-600之间（艺术类）
        # 2. 后面跟着4个数字：本科人数、本科累计、专科人数、专科累计

        i = 0
        while i < len(numbers) - 4:
            score = numbers[i]

            # 判断是否为有效分数
            is_valid_score = False
            if 200 <= score <= 750:
                is_valid_score = True
            elif 300 <= score <= 600:
                # 可能是艺术类合成总分
                is_valid_score = True

            if is_valid_score:
                # 检查后续数字是否合理
                count1 = numbers[i + 1]
                cum1 = numbers[i + 2]
                count2 = numbers[i + 3]
                cum2 = numbers[i + 4]

                # 累计人数应该大于等于本段人数
                if cum1 >= count1 and cum2 >= count2:
                    results.append({
                        '分数': score,
                        '本科人数': count1,
                        '本科累计人数': cum1,
                        '专科人数': count2,
                        '专科累计人数': cum2
                    })
                    i += 5  # 跳过已处理的5个数字
                    continue

            i += 1

        # 如果没有找到5个一组的，尝试3个一组（只有本科数据）
        if not results:
            i = 0
            while i < len(numbers) - 2:
                score = numbers[i]
                if 200 <= score <= 750:
                    results.append({
                        '分数': score,
                        '本科人数': numbers[i + 1],
                        '本科累计人数': numbers[i + 2],
                        '专科人数': 0,
                        '专科累计人数': 0
                    })
                    i += 3
                else:
                    i += 1

        return results

    # ==================== 投档情况提取 ====================

    def extract_admission_data(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """
        提取投档情况数据
        标准表格格式：院校代码、院校名称、专业组代码、计划数、投档人数、投档最低分、投档最低排位
        """
        logger.info(f"提取投档数据: {pdf_path.name}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_records = []

                for page in pdf.pages:
                    # 提取表格
                    tables = page.extract_tables()

                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        records = self._parse_admission_table(table)
                        all_records.extend(records)

                if not all_records:
                    logger.warning(f"  未提取到投档数据")
                    return None

                df = pd.DataFrame(all_records)

                # 过滤有效数据
                df = df[df['投档最低分'] > 0]
                df = df[df['院校代码'].str.len() == 5]

                logger.info(f"  ✓ 提取 {len(df)} 条投档记录")

                return df

        except Exception as e:
            logger.error(f"  提取失败: {e}")
            return None

    def _parse_admission_table(self, table: List[List]) -> List[Dict]:
        """解析投档情况表格"""
        records = []

        # 清理表格
        clean_table = []
        for row in table:
            if row and any(cell for cell in row if cell):
                clean_row = [self._clean_cell(cell) for cell in row]
                clean_table.append(clean_row)

        if len(clean_table) < 2:
            return records

        # 查找数据行
        for row in clean_table:
            # 跳过表头
            if self._is_admission_header(row):
                continue

            # 第一列应该是5位数字的院校代码
            if not row or not row[0]:
                continue

            college_code = row[0]
            if not college_code.isdigit() or len(college_code) != 5:
                continue

            # 提取各列数据
            record = self._extract_admission_row(row)
            if record:
                records.append(record)

        return records

    def _is_admission_header(self, row: List[str]) -> bool:
        """判断是否为投档表表头"""
        if not row:
            return False
        row_text = ' '.join(str(cell) for cell in row if cell)
        return '院校代码' in row_text or '院校名称' in row_text

    def _extract_admission_row(self, row: List[str]) -> Optional[Dict]:
        """提取单行投档数据"""
        if len(row) < 7:
            return None

        # 处理 "--" 表示无数据
        def parse_value(val):
            if val == '--' or val == '' or val is None:
                return 0
            try:
                return int(str(val).replace(',', ''))
            except:
                return 0

        return {
            '院校代码': row[0] if len(row) > 0 else '',
            '院校名称': row[1] if len(row) > 1 else '',
            '专业组代码': row[2] if len(row) > 2 else '',
            '计划数': parse_value(row[3]) if len(row) > 3 else 0,
            '投档人数': parse_value(row[4]) if len(row) > 4 else 0,
            '投档最低分': parse_value(row[5]) if len(row) > 5 else 0,
            '投档最低排位': parse_value(row[6]) if len(row) > 6 else 0
        }

    def _clean_cell(self, cell) -> str:
        """清理单元格内容"""
        if cell is None:
            return ""
        text = str(cell).strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    # ==================== 年份和类别识别 ====================

    def extract_year_and_category(self, filename: str) -> Tuple[Optional[int], str]:
        """从文件名提取年份和类别"""
        year_match = re.search(r'(\d{4})年', filename)
        year = int(year_match.group(1)) if year_match else None

        # 类别识别
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

    # ==================== 批量处理 ====================

    def process_all_pdfs(self) -> Dict[str, pd.DataFrame]:
        """处理所有PDF文件"""
        logger.info("=" * 70)
        logger.info("开始批量处理PDF文件")
        logger.info("=" * 70)

        all_admission = []
        all_scores = []

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
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
                df = self.extract_score_distribution(pdf_file)
                if df is not None and not df.empty:
                    df['年份'] = year
                    df['类别'] = category
                    all_scores.append(df)

        result = {}

        # 合并投档数据
        if all_admission:
            admission_df = pd.concat(all_admission, ignore_index=True)

            csv_path = self.output_dir / "admission_data_all.csv"
            admission_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 投档数据已保存: {csv_path}")

            # 统计
            logger.info(f"\n【投档数据统计】")
            logger.info(f"  总记录: {len(admission_df)} 条")
            logger.info(f"  覆盖院校: {admission_df['院校名称'].nunique()} 所")

            if '年份' in admission_df.columns:
                for year in sorted(admission_df['年份'].dropna().unique()):
                    year_df = admission_df[admission_df['年份'] == year]
                    logger.info(f"  {int(year)}年: {len(year_df)} 条")

            result['admission'] = admission_df

        # 合并分数段数据
        if all_scores:
            scores_df = pd.concat(all_scores, ignore_index=True)

            csv_path = self.output_dir / "score_distribution_all.csv"
            scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 分数段数据已保存: {csv_path}")

            # 统计
            logger.info(f"\n【分数段数据统计】")
            logger.info(f"  总记录: {len(scores_df)} 条")

            if '年份' in scores_df.columns and '类别' in scores_df.columns:
                for year in sorted(scores_df['年份'].dropna().unique()):
                    year_df = scores_df[scores_df['年份'] == year]
                    logger.info(f"  {int(year)}年:")
                    for cat in sorted(year_df['类别'].unique()):
                        cat_df = year_df[year_df['类别'] == cat]
                        logger.info(f"    {cat}: {len(cat_df)} 个分数段")

            result['score'] = scores_df

        return result

    def save_excel_report(self):
        """保存Excel格式报告"""
        admission_file = self.output_dir / "admission_data_all.csv"
        score_file = self.output_dir / "score_distribution_all.csv"

        excel_path = self.output_dir / "guangdong_gaokao_data.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 投档数据
            if admission_file.exists():
                df_admission = pd.read_csv(admission_file, encoding='utf-8-sig')
                df_admission.to_excel(writer, sheet_name='投档数据汇总', index=False)

                # 按年份分sheet
                if '年份' in df_admission.columns:
                    for year in sorted(df_admission['年份'].dropna().unique()):
                        year_df = df_admission[df_admission['年份'] == year]
                        sheet_name = f"{int(year)}年投档"
                        year_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            # 分数段数据
            if score_file.exists():
                df_score = pd.read_csv(score_file, encoding='utf-8-sig')
                df_score.to_excel(writer, sheet_name='分数段汇总', index=False)

        logger.info(f"\n✓ Excel报告已保存: {excel_path}")


def main():
    print("\n" + "=" * 70)
    print("  📊 通用PDF表格提取工具")
    print("=" * 70)
    print("\n支持格式:")
    print("  1. 分数段统计表 - 紧凑数字格式")
    print("  2. 投档情况表 - 标准表格格式")
    print("=" * 70)

    extractor = UniversalPDFExtractor(
        pdf_dir="data/raw_pdfs",
        output_dir="data/extracted_tables"
    )

    result = extractor.process_all_pdfs()
    extractor.save_excel_report()

    print("\n" + "=" * 70)
    print("  ✅ 提取完成！")
    print("=" * 70)
    print(f"\n输出目录: {extractor.output_dir.absolute()}")
    print("  - admission_data_all.csv     (投档数据)")
    print("  - score_distribution_all.csv (分数段数据)")
    print("  - guangdong_gaokao_data.xlsx (Excel报告)")


if __name__ == '__main__':
    main()