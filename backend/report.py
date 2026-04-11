"""
report.py - Generate a professional clinical-style PDF report using ReportLab
"""

import io
import base64
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak, KeepTogether
)

W, H = A4

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BLUE  = colors.HexColor('#1A2940')
MID_BLUE   = colors.HexColor('#2C3E50')
LIGHT_BLUE = colors.HexColor('#D6E4F0')
ACCENT     = colors.HexColor('#2563EB')
RED        = colors.HexColor('#DC2626')
GREEN      = colors.HexColor('#16A34A')
GRAY_BG    = colors.HexColor('#F7F9FC')
GRAY_LINE  = colors.HexColor('#BDC3C7')
WHITE      = colors.white


def _styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('RTitle', parent=base['Title'],
                                fontSize=22, textColor=DARK_BLUE,
                                spaceAfter=4, alignment=TA_CENTER),
        'subtitle': ParagraphStyle('RSub', parent=base['Normal'],
                                   fontSize=10, textColor=colors.HexColor('#555'),
                                   spaceAfter=2, alignment=TA_CENTER),
        'section': ParagraphStyle('RSection', parent=base['Heading2'],
                                  fontSize=12, textColor=WHITE,
                                  backColor=MID_BLUE, borderPad=(6, 6, 6, 8),
                                  spaceBefore=10, spaceAfter=4),
        'body': ParagraphStyle('RBody', parent=base['Normal'],
                               fontSize=9, leading=14,
                               textColor=colors.HexColor('#2C2C2C')),
        'small': ParagraphStyle('RSmall', parent=base['Normal'],
                                fontSize=8, textColor=colors.HexColor('#666')),
        'bold': ParagraphStyle('RBold', parent=base['Normal'],
                               fontSize=9, fontName='Helvetica-Bold'),
        'disclaimer': ParagraphStyle('RDisclaimer', parent=base['Normal'],
                                     fontSize=8, textColor=colors.HexColor('#888'),
                                     alignment=TA_CENTER),
    }


def _b64_to_rl_image(b64_str: str, width_cm: float, max_height_cm: float):
    """Convert a base64 data-URI to a ReportLab Image flowable. Returns None if invalid."""
    if not b64_str:
        return None
    try:
        if ',' in b64_str:
            b64_str = b64_str.split(',', 1)[1]
        raw = base64.b64decode(b64_str)
        buf = io.BytesIO(raw)
        img = RLImage(buf, width=width_cm * cm, height=max_height_cm * cm)
        img.hAlign = 'CENTER'
        return img
    except Exception:
        return None


def _meta_table(rows: list, styles) -> Table:
    tbl = Table(rows, colWidths=[5 * cm, 12 * cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  MID_BLUE),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  WHITE),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('BACKGROUND',    (0, 1), (0, -1),  LIGHT_BLUE),
        ('FONTNAME',      (0, 1), (0, -1),  'Helvetica-Bold'),
        ('BACKGROUND',    (1, 1), (1, -1),  GRAY_BG),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRAY_BG, colors.HexColor('#EAF0F8')]),
        ('GRID',          (0, 0), (-1, -1), 0.4, GRAY_LINE),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 7),
    ]))
    return tbl


def _image_grid(images: dict, result: dict) -> Table:
    """
    Build a 2×2 grid of analysis images:
    Original | Grad-CAM
    Mask     | Overlay
    """
    labels = {
        'original': 'Original MRI',
        'heatmap':  'Grad-CAM Heatmap',
        'mask':     'Tumor Mask',
        'overlay':  'Mask Overlay',
    }
    cell_w = 8.5   # cm
    cell_h = 6.5   # cm

    def cell(key):
        b64 = images.get(key)
        img = _b64_to_rl_image(b64, cell_w, cell_h) if b64 else None
        cap = Paragraph(labels[key],
                        ParagraphStyle('Cap', fontSize=7,
                                       textColor=colors.HexColor('#555'),
                                       alignment=TA_CENTER, spaceBefore=2))
        if img:
            return [img, cap]
        return Paragraph(f'<i>{labels[key]}<br/>Not available</i>',
                         ParagraphStyle('NA', fontSize=8,
                                        textColor=colors.gray,
                                        alignment=TA_CENTER))

    data = [
        [cell('original'), cell('heatmap')],
        [cell('mask'),     cell('overlay')],
    ]
    tbl = Table(data, colWidths=[cell_w * cm, cell_w * cm],
                rowHeights=[(cell_h + 0.6) * cm, (cell_h + 0.6) * cm])
    tbl.setStyle(TableStyle([
        ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',    (0, 0), (-1, -1), 0.3, GRAY_LINE),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _prob_bar_table(all_probs: dict, pred_type: str, styles) -> Table:
    """Horizontal probability bars for each tumor class."""
    BAR_W = 200   # points
    rows = [['Class', 'Probability', '']]
    for cls, prob in all_probs.items():
        label = cls.replace('_tumor', '').replace('_', ' ').title()
        pct   = f'{prob * 100:.1f}%'
        is_pred = (cls == pred_type)
        bar_color = ACCENT if is_pred else GRAY_LINE
        # Draw bar as a nested 1-row table
        filled = max(int(BAR_W * prob), 2)
        bar_tbl = Table([['']], colWidths=[filled], rowHeights=[10])
        bar_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bar_color),
            ('ROUNDEDCORNERS', [3]),
        ]))
        rows.append([
            Paragraph(f'<b>{label}</b>' if is_pred else label,
                      ParagraphStyle('CL', fontSize=9,
                                     fontName='Helvetica-Bold' if is_pred else 'Helvetica')),
            bar_tbl,
            Paragraph(f'<b>{pct}</b>' if is_pred else pct,
                      ParagraphStyle('PCT', fontSize=9, alignment=TA_RIGHT,
                                     fontName='Helvetica-Bold' if is_pred else 'Helvetica')),
        ])

    tbl = Table(rows, colWidths=[4 * cm, BAR_W * 0.0353 * cm, 2 * cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  MID_BLUE),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  WHITE),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [GRAY_BG, colors.HexColor('#EAF0F8')]),
        ('GRID',          (0, 0), (-1, -1), 0.3, GRAY_LINE),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 7),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return tbl


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(result: dict, filename: str = 'unknown.jpg') -> bytes:
    """
    Build and return a PDF report as bytes.
    `result` is the dict returned by run_pipeline().
    """
    buf    = io.BytesIO()
    styles = _styles()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=2 * cm,    bottomMargin=2 * cm,
    )

    story = []
    has_tumor  = result.get('tumor_detected', False)
    confidence = result.get('confidence', 0)
    tumor_type = result.get('tumor_type', 'N/A')
    type_conf  = result.get('type_confidence')
    all_probs  = result.get('all_probs', {})
    bbox       = result.get('bbox')
    now        = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

    # ── Cover / Header ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph('Brain Tumor Detection &amp; Classification', styles['title']))
    story.append(Paragraph('AI-Assisted Diagnostic Report  ·  ViT + ResNet-50', styles['subtitle']))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width='100%', thickness=2, color=ACCENT))
    story.append(Spacer(1, 0.4 * cm))

    # ── Report metadata ───────────────────────────────────────────────────────
    story.append(Paragraph('Report Information', styles['section']))
    story.append(Spacer(1, 0.2 * cm))
    story.append(_meta_table([
        ['Field', 'Value'],
        ['Image file',        filename],
        ['Report generated',  now],
        ['Detection model',   'Vision Transformer (ViT-Base/16) — Binary Classification'],
        ['Classification model', 'ResNet-50 — Glioma / Meningioma / Pituitary'],
        ['XAI method',        'Grad-CAM (layer4 of ResNet-50)'],
    ], styles))
    story.append(Spacer(1, 0.5 * cm))

    # ── Detection result ──────────────────────────────────────────────────────
    story.append(Paragraph('Detection Result', styles['section']))
    story.append(Spacer(1, 0.2 * cm))

    det_color = RED if has_tumor else GREEN
    det_label = 'TUMOR DETECTED' if has_tumor else 'NO TUMOR DETECTED'
    det_conf  = confidence if has_tumor else (1 - confidence)

    result_data = [
        ['Parameter', 'Result'],
        ['Detection outcome',    Paragraph(f'<font color="{det_color.hexval()}"><b>{det_label}</b></font>',
                                           styles['body'])],
        ['Detection confidence', f'{det_conf * 100:.1f}%'],
    ]
    if has_tumor:
        type_label = tumor_type.replace('_tumor', '').replace('_', ' ').title()
        result_data += [
            ['Tumor type',            type_label],
            ['Classification confidence', f'{(type_conf or 0) * 100:.1f}%'],
            ['Bounding box (x,y,w,h)', str(bbox) if bbox else 'N/A'],
        ]

    story.append(_meta_table(result_data, styles))
    story.append(Spacer(1, 0.5 * cm))

    # ── Class probabilities ───────────────────────────────────────────────────
    if has_tumor and all_probs:
        story.append(Paragraph('Classification Probabilities', styles['section']))
        story.append(Spacer(1, 0.2 * cm))
        story.append(_prob_bar_table(all_probs, tumor_type, styles))
        story.append(Spacer(1, 0.5 * cm))

    # ── Visual analysis ───────────────────────────────────────────────────────
    story.append(Paragraph('Visual Analysis', styles['section']))
    story.append(Spacer(1, 0.3 * cm))

    images = {
        'original': result.get('original'),
        'heatmap':  result.get('heatmap'),
        'mask':     result.get('mask'),
        'overlay':  result.get('overlay'),
    }
    story.append(_image_grid(images, result))
    story.append(Spacer(1, 0.5 * cm))

    # ── Clinical interpretation ───────────────────────────────────────────────
    story.append(Paragraph('Clinical Interpretation', styles['section']))
    story.append(Spacer(1, 0.2 * cm))

    tumor_info = {
        'glioma_tumor': (
            'Glioma',
            'Gliomas arise from glial cells and are the most common primary brain tumors. '
            'They vary widely in aggressiveness, ranging from slow-growing low-grade tumors '
            'to highly aggressive glioblastomas (Grade IV). Treatment typically involves '
            'surgery, radiation, and chemotherapy.'
        ),
        'meningioma_tumor': (
            'Meningioma',
            'Meningiomas develop from the meninges (the membranes surrounding the brain and '
            'spinal cord). They are usually benign and slow-growing, representing approximately '
            '36% of all primary brain tumors. Many are managed with observation; surgery or '
            'radiation is used when symptomatic.'
        ),
        'pituitary_tumor': (
            'Pituitary Adenoma',
            'Pituitary tumors affect the pituitary gland and can disrupt hormone regulation, '
            'leading to conditions such as Cushing\'s disease or acromegaly. The vast majority '
            'are benign adenomas. Treatment options include medication, surgery (transsphenoidal), '
            'and radiation therapy.'
        ),
    }

    if has_tumor:
        name, note = tumor_info.get(tumor_type, (tumor_type, ''))
        interp = (
            f'The ViT-based detection model identified <b>tumor tissue</b> with '
            f'<b>{confidence * 100:.1f}% confidence</b>. The ResNet-50 classifier '
            f'subsequently characterised the lesion as <b>{name}</b> '
            f'({(type_conf or 0) * 100:.1f}% confidence).<br/><br/>{note}'
        )
    else:
        interp = (
            f'The ViT-based detection model found <b>no evidence of a tumor</b> in this scan '
            f'(confidence: <b>{(1 - confidence) * 100:.1f}%</b>). The attention and Grad-CAM '
            f'maps show no localised abnormal activation patterns consistent with tumorous tissue.'
        )

    story.append(Paragraph(interp, styles['body']))
    story.append(Spacer(1, 0.6 * cm))

    # ── XAI explanation ───────────────────────────────────────────────────────
    story.append(KeepTogether([
        Paragraph('Explainability Methods', styles['section']),
        Spacer(1, 0.2 * cm),
        _meta_table([
            ['Method', 'Description'],
            ['ViT Attention Map',
             'Extracted from the last transformer block. Highlights image patches '
             'the model attends to during binary detection.'],
            ['Grad-CAM',
             'Gradient-weighted Class Activation Mapping applied to the final '
             'convolutional layer of ResNet-50. Shows which spatial regions most '
             'influenced the tumor-type classification.'],
            ['Tumor Mask',
             'Binary mask derived from the Grad-CAM heatmap by normalising and '
             'applying a threshold, followed by morphological smoothing.'],
            ['Overlay',
             'The binary mask blended onto the original MRI with 45% opacity '
             'to visualise the suspected tumor region in context.'],
        ], styles),
    ]))
    story.append(Spacer(1, 0.6 * cm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY_LINE))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        '⚠  DISCLAIMER: This report is generated by an AI research system and is intended '
        'for research and educational purposes only. It is NOT a substitute for professional '
        'medical diagnosis, advice, or treatment. Always consult a qualified medical '
        'professional for clinical decisions.',
        styles['disclaimer']
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
