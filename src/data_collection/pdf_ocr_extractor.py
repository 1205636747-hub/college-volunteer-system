"""
PDF图片表格OCR提取工具
用于提取广东省教育考试院PDF中的图片表格
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# 配置Tesseract路径（根据你的安装路径修改）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PDFOCRTableExtractor:
    """
    PDF图片表格OCR提取器
    专门处理广东省教育考试院PDF中的图片表格
    """

    def __init__(self, pdf_dir: str = "data/raw_pdfs", output_dir: str = "data/extracted_tables"):
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent
        self.pdf_dir = self.project_root / pdf_dir
        self.output_dir = self.project_root / output_dir
        self.temp_dir = self.project_root / 'data' / 'temp_images'

        # 创建目录
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PDF目录: {self.pdf_dir.absolute()}")
        logger.info(f"输出目录: {self.output_dir.absolute()}")
        logger.info(f"临时图片目录: {self.temp_dir.absolute()}")

    def pdf_to_images(self, pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
        """
        将PDF转换为图片列表

        Args:
            pdf_path: PDF文件路径
            dpi: 图片分辨率

        Returns:
            PIL Image列表
        """
        logger.info(f"  转换PDF为图片: {pdf_path.name}")

        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt='png',
                thread_count=2
            )
            logger.info(f"    ✓ 转换 {len(images)} 页")
            return images
        except Exception as e:
            logger.error(f"    ✗ PDF转换失败: {e}")
            return []

    def ocr_image_to_text(self, image: Image.Image) -> str:
        """
        对图片进行OCR识别，返回文本

        Args:
            image: PIL Image对象

        Returns:
            识别的文本
        """
        try:
            # 预处理图片提高识别率
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')

            # 使用中文语言包
            text = pytesseract.image_to_string(
                image,
                lang='chi_sim+eng',  # 简体中文+英文
                config='--psm 6'  # 假设为单个统一文本块
            )
            return text
        except Exception as e:
            logger.error(f"    OCR失败: {e}")
            return ""

    def extract_score_table_from_text(self, text: str, year: int = None) -> List[Dict]:
        """
        从OCR识别的文本中提取分数段表格

        格式示例：
        660 5 5 5
        659 0 5 0 5
        658 1 6 1 6
        """
        records = []
        lines = text.split('\n')

        # 分数段表格的特征：一行包含3-5个数字，第一个数字是3位数（分数）
        score_pattern = re.compile(r'^\s*(\d{3})\s+(\d+)\s+(\d+)(?:\s+(\d+)\s+(\d+))?')

        for line in lines:
            line = line.strip()
            match = score_pattern.match(line)
            if match:
                score = int(match.group(1))

                # 判断是否在合理分数范围内
                if 100 <= score <= 750:
                    # 本科数据
                    section_count = int(match.group(2))
                    cumulative_count = int(match.group(3))

                    records.append({
                        '分数': score,
                        '本段人数': section_count,
                        '累计人数': cumulative_count
                    })

        return records

    def extract_admission_table_from_text(self, text: str) -> List[Dict]:
        """
        从OCR识别的文本中提取投档表格

        格式示例：
        院校代码 院校名称 专业组代码 计划数 投档人数 投档最低分 投档最低排位
        10027 北京师范大学 206 1 1 576 2
        """
        records = []
        lines = text.split('\n')

        # 投档表格的特征：以5位数字开头
        admission_pattern = re.compile(
            r'^(\d{5})\s+([^\d]+?)\s+(\d{3,4})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
        )

        for line in lines:
            line = line.strip()
            match = admission_pattern.match(line)
            if match:
                record = {
                    '院校代码': match.group(1),
                    '院校名称': match.group(2).strip(),
                    '专业组代码': match.group(3),
                    '计划数': match.group(4),
                    '投档人数': match.group(5),
                    '投档最低分': match.group(6),
                    '投档最低排位': match.group(7)
                }
                records.append(record)

        return records

    def extract_score_distribution_ocr(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """
        使用OCR提取分数段统计表
        """
        logger.info(f"OCR提取分数段: {pdf_path.name}")

        # 提取年份和类别
        year, category = self.extract_year_and_category(pdf_path.name)

        # 转换PDF为图片
        images = self.pdf_to_images(pdf_path, dpi=250)
        if not images:
            return None

        all_records = []

        for page_num, image in enumerate(images, 1):
            # OCR识别
            text = self.ocr_image_to_text(image)

            if not text:
                continue

            # 提取分数段数据
            records = self.extract_score_table_from_text(text, year)

            if records:
                logger.info(f"    第{page_num}页: 提取 {len(records)} 个分数段")
                all_records.extend(records)

        if not all_records:
            # 尝试另一种识别策略
            all_records = self.extract_score_by_alternative_method(images)

        if not all_records:
            logger.warning(f"    ✗ 未提取到数据")
            return None

        # 去重并排序
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=['分数']).sort_values('分数', ascending=False)

        # 添加元数据
        df['年份'] = year
        df['类别'] = category

        logger.info(f"    ✓ 共提取 {len(df)} 个分数段")

        return df

    def extract_score_by_alternative_method(self, images: List[Image.Image]) -> List[Dict]:
        """
        备选方法：使用表格结构识别
        """
        records = []

        for image in images:
            # 尝试获取表格数据
            try:
                # 使用pytesseract获取带坐标的数据
                data = pytesseract.image_to_data(
                    image,
                    lang='chi_sim+eng',
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )

                # 按行分组
                rows = {}
                for i, text in enumerate(data['text']):
                    if text.strip():
                        y = data['top'][i]
                        x = data['left'][i]

                        # 按y坐标分组（同一行）
                        y_key = y // 20 * 20  # 容差20像素
                        if y_key not in rows:
                            rows[y_key] = []
                        rows[y_key].append({
                            'text': text.strip(),
                            'x': x,
                            'y': y
                        })

                # 分析每一行
                for y_key, row_data in sorted(rows.items()):
                    # 按x坐标排序
                    row_data.sort(key=lambda x: x['x'])
                    row_texts = [item['text'] for item in row_data]
                    row_str = ' '.join(row_texts)

                    # 查找分数模式
                    numbers = re.findall(r'\b\d+\b', row_str)
                    if len(numbers) >= 3:
                        first_num = int(numbers[0])
                        if 200 <= first_num <= 750:
                            records.append({
                                '分数': first_num,
                                '本段人数': int(numbers[1]) if len(numbers) > 1 else 0,
                                '累计人数': int(numbers[2]) if len(numbers) > 2 else 0
                            })

            except Exception as e:
                logger.debug(f"    备选方法失败: {e}")

        return records

    def extract_admission_data_ocr(self, pdf_path: Path) -> Optional[pd.DataFrame]:
        """
        使用OCR提取投档情况表
        """
        logger.info(f"OCR提取投档数据: {pdf_path.name}")

        year, category = self.extract_year_and_category(pdf_path.name)

        # 转换PDF为图片
        images = self.pdf_to_images(pdf_path, dpi=250)
        if not images:
            return None

        all_records = []

        for page_num, image in enumerate(images, 1):
            # OCR识别
            text = self.ocr_image_to_text(image)

            if not text:
                continue

            # 提取投档数据
            records = self.extract_admission_table_from_text(text)

            if records:
                logger.info(f"    第{page_num}页: 提取 {len(records)} 条记录")
                all_records.extend(records)

        if not all_records:
            logger.warning(f"    ✗ 未提取到数据")
            return None

        df = pd.DataFrame(all_records)

        # 清理数据
        if '院校代码' in df.columns:
            df = df[df['院校代码'].apply(lambda x: str(x).isdigit() and len(str(x)) == 5)]

        # 转换数据类型
        numeric_cols = ['计划数', '投档人数', '投档最低分', '投档最低排位']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '投档最低分' in df.columns:
            df = df[df['投档最低分'] > 0]

        # 添加元数据
        df['年份'] = year
        df['类别'] = category

        logger.info(f"    ✓ 共提取 {len(df)} 条记录")

        return df

    def extract_year_and_category(self, filename: str) -> Tuple[Optional[int], str]:
        """从文件名提取年份和类别"""
        year_match = re.search(r'(\d{4})年', filename)
        year = int(year_match.group(1)) if year_match else None

        if '历史' in filename:
            category = '历史类'
        elif '物理' in filename:
            category = '物理类'
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
        else:
            category = '其他'

        return year, category

    def process_all_pdfs(self, use_ocr: bool = True) -> Dict[str, pd.DataFrame]:
        """
        处理所有PDF文件

        Args:
            use_ocr: 是否使用OCR模式（默认True）
        """
        logger.info("=" * 70)
        logger.info("开始批量处理PDF文件（OCR模式）")
        logger.info("=" * 70)

        all_admission = []
        all_scores = []

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件\n")

        success_count = 0
        fail_count = 0

        for i, pdf_file in enumerate(pdf_files, 1):
            filename = pdf_file.name
            logger.info(f"[{i}/{len(pdf_files)}] 处理: {filename}")

            if '投档' in filename:
                df = self.extract_admission_data_ocr(pdf_file)
                if df is not None and not df.empty:
                    all_admission.append(df)
                    success_count += 1
                else:
                    fail_count += 1

            elif '分数段' in filename or '统计表' in filename:
                df = self.extract_score_distribution_ocr(pdf_file)
                if df is not None and not df.empty:
                    all_scores.append(df)
                    success_count += 1
                else:
                    fail_count += 1

            print()

        logger.info("=" * 70)
        logger.info(f"处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        logger.info("=" * 70)

        result = {}

        # 保存投档数据
        if all_admission:
            admission_df = pd.concat(all_admission, ignore_index=True)

            csv_path = self.output_dir / "admission_data_ocr.csv"
            admission_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 投档数据已保存: {csv_path}")

            # Excel格式
            excel_path = self.output_dir / "admission_data_ocr.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                admission_df.to_excel(writer, sheet_name='全部数据', index=False)

                if '年份' in admission_df.columns:
                    for year in admission_df['年份'].dropna().unique():
                        year_df = admission_df[admission_df['年份'] == year]
                        year_df.to_excel(writer, sheet_name=f"{int(year)}年", index=False)

            logger.info(f"✓ Excel已保存: {excel_path}")

            # 统计
            logger.info(f"\n投档数据统计:")
            logger.info(f"  总记录: {len(admission_df)} 条")
            if '年份' in admission_df.columns:
                for year in sorted(admission_df['年份'].dropna().unique()):
                    count = len(admission_df[admission_df['年份'] == year])
                    logger.info(f"  {int(year)}年: {count} 条")

            result['admission'] = admission_df

        # 保存分数段数据
        if all_scores:
            scores_df = pd.concat(all_scores, ignore_index=True)

            csv_path = self.output_dir / "score_distribution_ocr.csv"
            scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"\n✓ 分数段数据已保存: {csv_path}")

            excel_path = self.output_dir / "score_distribution_ocr.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                scores_df.to_excel(writer, sheet_name='全部数据', index=False)

                if '年份' in scores_df.columns and '类别' in scores_df.columns:
                    for year in scores_df['年份'].dropna().unique():
                        year_df = scores_df[scores_df['年份'] == year]
                        for cat in year_df['类别'].unique():
                            cat_df = year_df[year_df['类别'] == cat]
                            if not cat_df.empty:
                                sheet_name = f"{int(year)}年{cat}"[:31]
                                cat_df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(f"✓ Excel已保存: {excel_path}")

            logger.info(f"\n分数段数据统计:")
            logger.info(f"  总记录: {len(scores_df)} 条")

            result['score'] = scores_df

        # 清理临时文件
        self.cleanup_temp_files()

        return result

    def cleanup_temp_files(self):
        """清理临时图片文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"\n✓ 临时文件已清理")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  📊 PDF图片表格OCR提取工具")
    print("=" * 70)
    print("\n说明:")
    print("  本工具使用OCR技术识别PDF中的图片表格")
    print("  适用于广东省教育考试院发布的PDF文件")
    print("\n依赖:")
    print("  - pdf2image (PDF转图片)")
    print("  - pytesseract (OCR识别)")
    print("  - Tesseract-OCR (需单独安装)")
    print("\n" + "=" * 70)

    # 检查Tesseract是否安装
    try:
        version = pytesseract.get_tesseract_version()
        print(f"\n✓ Tesseract版本: {version}")
    except Exception as e:
        print(f"\n✗ Tesseract未正确安装: {e}")
        print("请从以下地址下载安装:")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        print("安装时请勾选中文语言包")
        return

    # 创建提取器
    extractor = PDFOCRTableExtractor(
        pdf_dir="data/raw_pdfs",
        output_dir="data/extracted_tables"
    )

    # 处理所有PDF
    result = extractor.process_all_pdfs(use_ocr=True)

    print("\n" + "=" * 70)
    print("  ✅ 提取完成！")
    print("=" * 70)
    print(f"\n输出目录: {extractor.output_dir.absolute()}")
    print("  - admission_data_ocr.csv/xlsx")
    print("  - score_distribution_ocr.csv/xlsx")


if __name__ == '__main__':
    main()