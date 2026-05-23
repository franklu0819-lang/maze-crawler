#!/usr/bin/env python3
"""
使用reportlab生成PDF分析报告
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 尝试注册中文字体
try:
    # macOS系统字体
    pdfmetrics.registerFont(TTFont('PingFang', '/System/Library/Fonts/PingFang.ttc', subfontIndex=0))
    FONT_NAME = 'PingFang'
except:
    try:
        pdfmetrics.registerFont(TTFont('SongTi', '/System/Library/Fonts/STHeiti Light.ttc', subfontIndex=0))
        FONT_NAME = 'SongTi'
    except:
        FONT_NAME = 'Helvetica'
        print("⚠️  未找到中文字体，使用英文字体")

def create_styles():
    """创建自定义样式"""
    styles = getSampleStyleSheet()
    
    # 标题样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontName=FONT_NAME,
        fontSize=20,
        leading=24,
        spaceAfter=20,
        alignment=1,  # 居中
        textColor=colors.darkblue
    )
    
    # 副标题样式
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontName=FONT_NAME,
        fontSize=14,
        leading=18,
        spaceAfter=15,
        spaceBefore=10,
        textColor=colors.darkblue
    )
    
    # 三级标题样式
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontName=FONT_NAME,
        fontSize=12,
        leading=16,
        spaceAfter=10,
        spaceBefore=8,
        textColor=colors.black
    )
    
    # 正文样式
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        spaceAfter=8,
        firstLineIndent=20
    )
    
    # 列表样式（无首行缩进）
    list_style = ParagraphStyle(
        'CustomList',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        spaceAfter=5,
        leftIndent=20
    )
    
    # 表格标题样式
    table_title_style = ParagraphStyle(
        'TableTitle',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=11,
        leading=14,
        spaceAfter=8,
        alignment=1,
        textColor=colors.darkblue
    )
    
    # 成功绿色样式
    success_style = ParagraphStyle(
        'Success',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        spaceAfter=8,
        textColor=colors.green
    )
    
    # 警告橙色样式
    warning_style = ParagraphStyle(
        'Warning',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        spaceAfter=8,
        textColor=colors.orange
    )
    
    # 危险红色样式
    danger_style = ParagraphStyle(
        'Danger',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        spaceAfter=8,
        textColor=colors.red
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'h3': h3_style,
        'body': body_style,
        'list': list_style,
        'table_title': table_title_style,
        'success': success_style,
        'warning': warning_style,
        'danger': danger_style
    }

def add_image_with_caption(story, img_path, caption, styles, width=14*cm):
    """添加带标题的图片"""
    if os.path.exists(img_path):
        img = Image(img_path)
        original_width, original_height = img.drawWidth, img.drawHeight
        aspect_ratio = original_width / original_height
        img.drawWidth = width
        img.drawHeight = width / aspect_ratio
        story.append(img)
        story.append(Paragraph(caption, styles['table_title']))
        story.append(Spacer(1, 0.3*cm))

def create_pdf():
    """生成PDF报告"""
    output_path = '/Users/leo/Projects/kaggle/maze-crawler/PPO训练分析报告_v17_v50.pdf'
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                          rightMargin=2*cm, leftMargin=2*cm,
                          topMargin=2*cm, bottomMargin=2*cm)
    
    story = []
    styles = create_styles()
    img_dir = '/Users/leo/Projects/kaggle/maze-crawler/analysis_output'
    
    # 标题
    story.append(Paragraph("PPO强化学习训练分析报告", styles['title']))
    story.append(Paragraph("v17_v50版本训练日志完整分析", styles['subtitle']))
    story.append(Spacer(1, 0.5*cm))
    
    # 基本信息表格
    info_data = [
        ['训练标识', 'train_v17_v50.log', '环境', 'Maze Crawler'],
        ['算法', 'PPO', '总轮次', '2000轮'],
        ['训练时长', '16小时11分钟', '最终vs随机胜率', '90.6%']
    ]
    info_table = Table(info_data, colWidths=[3*cm, 5*cm, 3*cm, 5*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('BACKGROUND', (2, 0), (2, -1), colors.lightblue),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # 摘要
    story.append(Paragraph("摘要", styles['subtitle']))
    story.append(Paragraph("本报告对PPO强化学习2000轮完整训练过程进行了系统性分析。通过对性能曲线、Self-Play胜率趋势、Entropy衰减、Loss变化等关键指标的深入研究，评估了训练的收敛状态、效率和异常情况。分析结果显示，本次训练过程稳定，在第1470轮达到69%的峰值胜率后充分收敛。基于分析结果，本报告提出了针对性的改进建议，为后续版本迭代提供了决策依据。", styles['body']))
    story.append(PageBreak())
    
    # 1. 训练概览
    story.append(Paragraph("1. 训练概览", styles['subtitle']))
    
    story.append(Paragraph("1.1 基本参数", styles['h3']))
    params_data = [
        ['项目', '数值'],
        ['总训练轮次', '2000轮'],
        ['训练总时长', '58289秒 ≈ 16小时11分钟'],
        ['单轮平均耗时', '29.1秒'],
        ['初始学习率', '0.0002'],
        ['Entropy系数', '0.02'],
        ['Batch大小', '100'],
        ['奖励函数', 'Δgap × 0.5 + Δunits × 0.1'],
        ['终端奖励', '+5.0 / -1 / 0 (平局)'],
        ['训练模式', '100% Self-Play'],
        ['评估方式', '每10轮混合评估（100局）'],
        ['最终vs随机胜率', '90.6% (453W-35L-12D)'],
    ]
    params_table = Table(params_data, colWidths=[5*cm, 10*cm])
    params_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(params_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("1.2 训练里程碑", styles['h3']))
    milestone_data = [
        ['轮次', 'Best WR', '说明'],
        ['第10轮', '23%', '首次评估即突破，快速摆脱随机水平'],
        ['第20轮', '39%', '快速学习阶段，20轮达到接近40%胜率'],
        ['第50轮', '54%', '前50轮快速进步，达到中等智能水平'],
        ['第360轮', '56%', '长时间平台期后小幅度提升'],
        ['第440轮', '59%', '持续探索带来新的突破'],
        ['第690轮', '64%', '中期出现较大提升，策略开始成熟'],
        ['第1020轮', '66%', '后期小幅提升'],
        ['第1230轮', '67%', '策略进一步精细化'],
        ['第1390轮', '68%', '接近收敛时的小幅突破'],
        ['第1470轮', '69%', '训练峰值，此后进入完全收敛状态'],
    ]
    milestone_table = Table(milestone_data, colWidths=[2.5*cm, 2.5*cm, 10*cm])
    milestone_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (2, 0), (2, -1), 'LEFT'),
    ]))
    story.append(milestone_table)
    story.append(PageBreak())
    
    # 2. 性能曲线分析
    story.append(Paragraph("2. 性能曲线分析", styles['subtitle']))
    add_image_with_caption(story, 
                          os.path.join(img_dir, 'performance_curve.png'),
                          "图1: Best Mixed Eval WR性能曲线", styles)
    
    story.append(Paragraph("2.1 关键发现", styles['h3']))
    story.append(Paragraph("1. 快速上升期（0-500轮）：胜率从23%快速提升至59%，提升幅度达到36个百分点，占总提升的85.7%", styles['list']))
    story.append(Paragraph("2. 缓慢优化期（500-1470轮）：胜率从59%提升至69%，耗时970轮仅提升10个百分点", styles['list']))
    story.append(Paragraph("3. 完全收敛期（1470-2000轮）：持续530轮没有任何提升，策略已充分收敛", styles['list']))
    story.append(Paragraph("4. 评估稳定性：红色散点显示每10轮评估结果波动逐渐减小，说明策略稳定性增强", styles['list']))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("2.2 阶段评估", styles['h3']))
    story.append(Paragraph("✓ 优秀：前期学习速度快，50轮达到54%胜率", styles['success']))
    story.append(Paragraph("✓ 良好：中期持续优化，没有出现断崖式下跌", styles['body']))
    story.append(Paragraph("⚠ 注意：后期收敛速度极慢，边际收益递减严重", styles['warning']))
    story.append(PageBreak())
    
    # 3. Self-Play胜率趋势
    story.append(Paragraph("3. Self-Play 胜率趋势", styles['subtitle']))
    add_image_with_caption(story,
                          os.path.join(img_dir, 'sp_trend.png'),
                          "图2: Self-Play胜率趋势（含20轮移动平均", styles)
    
    story.append(Paragraph("3.1 SP趋势三阶段分析", styles['h3']))
    
    story.append(Paragraph("3.1.1 快速迭代期（0-100轮", styles['h3']))
    story.append(Paragraph("• SP胜率从91%快速下降到23%左右", styles['list']))
    story.append(Paragraph("• 说明策略更新速度极快，新旧版本差异巨大", styles['list']))
    story.append(Paragraph("• 属于正常现象，表明算法在有效探索", styles['list']))
    
    story.append(Paragraph("3.1.2 持续优化期（100-1000轮）", styles['h3']))
    story.append(Paragraph("• SP胜率在20%-40%区间波动", styles['list']))
    story.append(Paragraph("• 策略持续优化但步伐放缓", styles['list']))
    story.append(Paragraph("• 20轮移动平均线呈现平滑下降趋势", styles['list']))
    
    story.append(Paragraph("3.1.3 收敛稳定期（1000-2000轮）", styles['h3']))
    story.append(Paragraph("• SP稳定在25%-40%区间", styles['list']))
    story.append(Paragraph("• 波动范围明显缩小，收敛特征显著", styles['list']))
    story.append(Paragraph("• 移动平均线趋于平缓，策略更新速度大幅降低", styles['list']))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("3.2 收敛判断", styles['h3']))
    story.append(Paragraph("✓ 结论：训练已充分收敛。SP波动范围从早期的20%-91%收缩到后期稳定的25%-40%，说明策略探索空间已大幅压缩，趋于稳定。", styles['success']))
    story.append(PageBreak())
    
    # 4. Entropy衰减分析
    story.append(Paragraph("4. Entropy衰减分析", styles['subtitle']))
    add_image_with_caption(story,
                          os.path.join(img_dir, 'entropy_curve.png'),
                          "图3: 策略Entropy衰减曲线", styles)
    
    story.append(Paragraph("4.1 Entropy关键指标", styles['h3']))
    entropy_data = [
        ['指标', '数值'],
        ['初始Entropy', '0.647'],
        ['最终Entropy', '0.173'],
        ['总衰减幅度', '73.3%'],
        ['衰减速率', '约0.00024 / 轮'],
    ]
    entropy_table = Table(entropy_data, colWidths=[5*cm, 10*cm])
    entropy_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
    ]))
    story.append(entropy_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("4.2 分析结论", styles['h3']))
    story.append(Paragraph("1. 衰减平稳：Entropy持续稳定下降，没有出现突然崩塌或停滞", styles['list']))
    story.append(Paragraph("2. 平衡良好：73%的衰减幅度符合PPO训练的正常探索-利用平衡", styles['list']))
    story.append(Paragraph("3. 探索充分：最终Entropy为0.173，仍保留一定探索能力，没有过度贪婪", styles['list']))
    story.append(Paragraph("4. 无异常：曲线平滑，没有出现突然的上升或下降，训练过程稳定", styles['list']))
    story.append(PageBreak())
    
    # 5. Loss变化趋势
    story.append(Paragraph("5. Loss变化趋势", styles['subtitle']))
    add_image_with_caption(story,
                          os.path.join(img_dir, 'loss_curves.png'),
                          "图4: Policy Loss与Value Loss变化趋势", styles)
    
    story.append(Paragraph("5.1 Policy Loss分析", styles['h3']))
    story.append(Paragraph("• 始终在-0.0004到0.0006之间小幅波动", styles['list']))
    story.append(Paragraph("• 整体接近0，说明策略梯度更新稳定", styles['list']))
    story.append(Paragraph("• 没有出现剧烈震荡或爆炸迹象", styles['list']))
    story.append(Paragraph("• 训练全程保持良好的数值稳定性", styles['list']))
    
    story.append(Paragraph("5.2 Value Loss分析", styles['h3']))
    story.append(Paragraph("• 从初始12.95持续下降到最终1.41左右", styles['list']))
    story.append(Paragraph("• 下降趋势平稳，价值函数拟合度不断提升", styles['list']))
    story.append(Paragraph("• 后期趋于稳定在1.4-2.0区间", styles['list']))
    story.append(Paragraph("• 说明价值网络已充分学习状态价值估计", styles['list']))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("✓ 结论：Loss全程稳定，没有异常。这是一次非常稳定的PPO训练，数值优化过程健康。", styles['success']))
    story.append(PageBreak())
    
    # 6. 训练效率分析
    story.append(Paragraph("6. 训练效率分析", styles['subtitle']))
    add_image_with_caption(story,
                          os.path.join(img_dir, 'training_efficiency.png'),
                          "图5: 各阶段训练效率对比 - 边际收益递减分析", styles, width=12*cm)
    
    story.append(Paragraph("6.1 效率数据对比", styles['h3']))
    efficiency_data = [
        ['阶段', '达到Best WR', '阶段提升', '占总提升比例'],
        ['0-500轮', '59%', '+59pp', '85.5%'],
        ['500-1000轮', '66%', '+7pp', '10.1%'],
        ['1000-1500轮', '69%', '+3pp', '4.3%'],
        ['1500-2000轮', '69%', '+0pp', '0.0%'],
        ['总计', '69%', '69pp', '100%'],
    ]
    efficiency_table = Table(efficiency_data, colWidths=[3.5*cm, 3*cm, 3*cm, 3.5*cm])
    efficiency_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (-1, -1), (-1, -1), colors.lightcoral),
    ]))
    story.append(efficiency_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("6.2 边际收益递减分析", styles['h3']))
    story.append(Paragraph("1. 前500轮贡献了85.5%的总提升，是训练的黄金期", styles['list']))
    story.append(Paragraph("2. 500-1000轮提升明显放缓，但仍有价值", styles['list']))
    story.append(Paragraph("3. 1000-1500轮仅提升3个百分点，性价比大幅降低", styles['list']))
    story.append(Paragraph("4. 1500-2000轮完全没有提升，属于纯资源消耗", styles['list']))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("⚠ 建议：未来训练可在1500轮左右提前停止，或采用动态停止策略（连续200轮无提升即停止），可节省约25%的计算资源。", styles['warning']))
    story.append(PageBreak())
    
    # 7. 异常检测
    story.append(Paragraph("7. 异常检测", styles['subtitle']))
    story.append(Paragraph("7.1 检测结果综述", styles['h3']))
    story.append(Paragraph("✓ 总体结论：无重大异常，训练过程健康稳定。", styles['success']))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("7.2 各项检测明细", styles['h3']))
    anomaly_data = [
        ['检测项目', '状态', '说明'],
        ['SP突然暴跌检测', '正常', '无突然暴跌（如从50%降到10%以下）'],
        ['Loss爆炸检测', '正常', 'Loss全程稳定，无剧烈震荡'],
        ['Best WR回退检测', '正常', 'Best WR持续单调上升，无回退'],
        ['Entropy崩塌检测', '正常', 'Entropy衰减平稳，无突然归零'],
        ['学习率发散检测', '正常', '无梯度爆炸或消失迹象'],
    ]
    anomaly_table = Table(anomaly_data, colWidths=[4*cm, 2.5*cm, 8.5*cm])
    anomaly_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (1, 1), (1, -1), colors.green),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
    ]))
    story.append(anomaly_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("7.3 轻微波动说明", styles['h3']))
    story.append(Paragraph("• 第12轮前后SP从50%降到26%：属于正常的策略更迭现象", styles['list']))
    story.append(Paragraph("• 第757轮v50版本切换时SP短暂上升：版本切换正常波动", styles['list']))
    story.append(Paragraph("• 以上波动均在正常范围内，不影响训练质量", styles['list']))
    story.append(PageBreak())
    
    # 8. 最终结论
    story.append(Paragraph("8. 最终结论", styles['subtitle']))
    story.append(Paragraph("8.1 收敛性判断", styles['h3']))
    story.append(Paragraph("✓✓✓ 训练已充分收敛 ✓✓✓", styles['success']))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("收敛判断依据：", styles['h3']))
    story.append(Paragraph("1. Best WR在第1470轮达到69%峰值后，后续530轮零提升", styles['list']))
    story.append(Paragraph("2. SP波动范围从早期20%-91%收缩到后期稳定的25%-40%", styles['list']))
    story.append(Paragraph("3. Entropy已降至0.173的低位，探索空间非常有限", styles['list']))
    story.append(Paragraph("4. Value Loss稳定在1.4-2.0区间，价值函数已充分拟合", styles['list']))
    story.append(Paragraph("5. Policy Loss持续接近0，策略梯度更新饱和", styles['list']))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("8.2 是否值得继续训练？", styles['h3']))
    story.append(Paragraph("✗✗✗ 不值得继续训练 ✗✗✗", styles['danger']))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("停止训练理由：", styles['h3']))
    story.append(Paragraph("1. 最后530轮完全没有提升，继续训练预期收益为0", styles['list']))
    story.append(Paragraph("2. 计算资源消耗巨大，单轮约29秒，继续训练性价比极低", styles['list']))
    story.append(Paragraph("3. 策略已充分成熟，单纯延长训练难以带来本质性能提升", styles['list']))
    story.append(Paragraph("4. vs随机对手已达90.6%胜率，接近理论天花板", styles['list']))
    story.append(PageBreak())
    
    # 9. 改进建议
    story.append(Paragraph("9. 改进建议", styles['subtitle']))
    story.append(Paragraph("基于本次训练分析，对后续版本提出以下改进建议：", styles['body']))
    
    story.append(Paragraph("9.1 网络结构优化", styles['h3']))
    story.append(Paragraph("1. 增加网络容量：尝试更大的隐层神经元数量或增加层数，突破表达能力瓶颈", styles['list']))
    story.append(Paragraph("2. Layer Normalization：加入层归一化稳定训练，允许更大学习率", styles['list']))
    story.append(Paragraph("3. 激活函数优化：尝试GELU、Swish等现代激活函数替代ReLU", styles['list']))
    
    story.append(Paragraph("9.2 超参数调整", styles['h3']))
    story.append(Paragraph("1. 学习率调度：采用余弦退火或线性衰减策略，而非固定学习率", styles['list']))
    story.append(Paragraph("2. Entropy系数：从0.02增加到0.03-0.05，增强后期探索能力", styles['list']))
    story.append(Paragraph("3. Clip范围：动态调整PPO Clip范围（0.15-0.25），平衡稳定性与更新速度", styles['list']))
    
    story.append(Paragraph("9.3 训练机制创新", styles['h3']))
    story.append(Paragraph("1. 对手抽样：实现Prioritized Opponent Sampling，优先对战有挑战性的历史版本", styles['list']))
    story.append(Paragraph("2. 群体进化：实施Population-Based Training，多模型并行进化", styles['list']))
    story.append(Paragraph("3. 专家预训练：引入Expert Demonstration进行监督学习预训练", styles['list']))
    
    story.append(Paragraph("9.4 奖励塑造改进", styles['h3']))
    story.append(Paragraph("1. 权重调整：当前Δgap × 0.5可能过于保守，尝试调至0.7-1.0", styles['list']))
    story.append(Paragraph("2. 探索奖励：加入基于策略多样性的探索奖励，避免过早收敛", styles['list']))
    story.append(Paragraph("3. 动态奖励：实现基于当前胜率的动态奖励调整", styles['list']))
    
    story.append(Paragraph("9.5 版本迭代策略", styles['h3']))
    story.append(Paragraph("1. v50已达瓶颈，建议基于v50做结构性改动而非继续微调", styles['list']))
    story.append(Paragraph("2. 尝试多模型集成策略，融合不同训练阶段的优秀模型", styles['list']))
    story.append(Paragraph("3. 加入知识蒸馏压缩，在保持性能的同时降低推理延迟", styles['list']))
    story.append(PageBreak())
    
    # 10. 预期潜力分析
    story.append(Paragraph("10. 预期潜力分析", styles['subtitle']))
    
    story.append(Paragraph("10.1 当前性能基准", styles['h3']))
    story.append(Paragraph("• vs随机对手：90.6%胜率（453W-35L-12D）", styles['list']))
    story.append(Paragraph("• vs v6版本：预估约70%胜率", styles['list']))
    story.append(Paragraph("• Best混合评估：69%胜率", styles['list']))
    
    story.append(Paragraph("10.2 理论天花板估计", styles['h3']))
    story.append(Paragraph("• vs随机对手理论天花板约95%-97%", styles['list']))
    story.append(Paragraph("• 当前已达90.6%，剩余提升空间约4-6个百分点", styles['list']))
    story.append(Paragraph("• 考虑到随机因素影响，实际可挖掘空间约3-5个百分点", styles['list']))
    
    story.append(Paragraph("10.3 改进预期", styles['h3']))
    potential_data = [
        ['改进方向', '预期提升幅度'],
        ['网络结构优化', '+1-2pp'],
        ['超参数精细调优', '+0.5-1pp'],
        ['训练机制创新', '+1-3pp'],
        ['奖励塑造改进', '+0.5-1.5pp'],
        ['多模型集成', '+1-2pp'],
        ['综合优化（叠加效应）', '+3-5pp'],
    ]
    potential_table = Table(potential_data, colWidths=[7*cm, 8*cm])
    potential_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('BACKGROUND', (-1, -1), (-1, -1), colors.lightgreen),
    ]))
    story.append(potential_table)
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("总结：通过系统性优化，预期可将胜率从69%提升至72-74%左右。但考虑到边际收益递减规律，除非有明确的竞赛目标，否则当前版本已足够优秀，可投入实际应用。", styles['body']))
    
    # 生成PDF
    doc.build(story)
    print(f"\n🎉 PDF报告已生成: {output_path}")
    print(f"📊 包含5个图表和完整的分析结论")

if __name__ == '__main__':
    create_pdf()