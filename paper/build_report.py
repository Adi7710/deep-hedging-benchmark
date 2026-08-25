from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_JUSTIFY

ACC = colors.HexColor("#1a3d6d"); MUT = colors.HexColor("#5a6472")
LGT = colors.HexColor("#f2f4f7"); RULE = colors.HexColor("#c9d1dc")
OK = colors.HexColor("#1d6b3a"); WARN = colors.HexColor("#a3341f")

ss = getSampleStyleSheet()
S = {}
S['title']  = ParagraphStyle('t', fontName='Helvetica-Bold', fontSize=21, leading=26, textColor=ACC, spaceAfter=5)
S['sub']    = ParagraphStyle('s', fontName='Helvetica', fontSize=11.5, leading=15, textColor=MUT, spaceAfter=3)
S['h1']     = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=15, leading=19, textColor=ACC, spaceBefore=15, spaceAfter=7)
S['h2']     = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=11.8, leading=15, textColor=colors.HexColor("#22303f"), spaceBefore=11, spaceAfter=4)
S['h3']     = ParagraphStyle('h3', fontName='Helvetica-BoldOblique', fontSize=10.2, leading=13, textColor=MUT, spaceBefore=8, spaceAfter=3)
S['body']   = ParagraphStyle('b', fontName='Helvetica', fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6)
S['bullet'] = ParagraphStyle('bu', parent=S['body'], leftIndent=12, bulletIndent=3, spaceAfter=3)
S['code']   = ParagraphStyle('c', fontName='Courier', fontSize=8.1, leading=10.6, textColor=colors.HexColor("#12233a"),
                             backColor=LGT, borderPadding=(6,6,6,6), leftIndent=3, rightIndent=3, spaceBefore=4, spaceAfter=7)
S['note']   = ParagraphStyle('n', parent=S['body'], leftIndent=10, borderPadding=(6,7,6,7),
                             backColor=colors.HexColor("#fbf7e8"), spaceBefore=5, spaceAfter=8)
S['cap']    = ParagraphStyle('cap', fontName='Helvetica-Oblique', fontSize=8.3, leading=11, textColor=MUT, spaceAfter=8)
S['cell']   = ParagraphStyle('ce', fontName='Helvetica', fontSize=8.4, leading=11)
S['cellb']  = ParagraphStyle('cb', fontName='Helvetica-Bold', fontSize=8.4, leading=11, textColor=colors.white)
S['cellm']  = ParagraphStyle('cm', fontName='Courier', fontSize=8.0, leading=10.5)

def esc(t): return t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def P(t, st='body'): return Paragraph(t, S[st])
def H1(t): return Paragraph(t, S['h1'])
def H2(t): return Paragraph(t, S['h2'])
def H3(t): return Paragraph(t, S['h3'])
def CODE(t): return Paragraph(esc(t).replace('\n','<br/>').replace(' ','&nbsp;'), S['code'])
def NOTE(t): return Paragraph(t, S['note'])
def BUL(items):
    return [Paragraph(f'<bullet>-</bullet>{t}', S['bullet']) for t in items]
def TBL(rows, widths, mono_cols=()):
    data=[[Paragraph(esc(c), S['cellb']) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(esc(c), S['cellm'] if j in mono_cols else S['cell']) for j,c in enumerate(r)])
    t=Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),ACC),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGT]),
        ('GRID',(0,0),(-1,-1),0.4,RULE),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return t

W,Hh = A4
def deco(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5)
    canvas.line(20*mm, 16*mm, W-20*mm, 16*mm)
    canvas.setFont('Helvetica', 7.6); canvas.setFillColor(MUT)
    canvas.drawString(20*mm, 11*mm, "Deep Hedging Benchmark - Technical Report and Research Plan")
    canvas.drawRightString(W-20*mm, 11*mm, f"page {doc.page}")
    canvas.restoreState()

import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "technical-report.pdf")
doc = BaseDocTemplate(_OUT, pagesize=A4,
                      leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=22*mm,
                      title="Deep Hedging Benchmark — Technical Report and Research Plan",
                      author="Aditya Bhatia")
doc.addPageTemplates([PageTemplate(id='n', frames=[Frame(doc.leftMargin, doc.bottomMargin,
                      doc.width, doc.height, id='f')], onPage=deco)])

E=[]
import os as _o, sys as _s
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
import report_content as content
E = content.build(P,H1,H2,H3,CODE,NOTE,BUL,TBL,Spacer,PageBreak,S,mm,doc)
doc.build(E)
print("written", _OUT)
