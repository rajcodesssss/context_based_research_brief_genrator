#!/usr/bin/env python3
"""
PDF Converter Module for Research Briefs
Converts JSON research data to formatted PDF reports and full research papers
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, blue, darkblue, green, darkgreen
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib import colors
from urllib.parse import urlparse
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

class ResearchPDFGenerator:
    """Generate formatted PDF reports and full research papers from research brief data"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the PDF"""
        # Define custom style names to avoid conflicts
        custom_styles = [
            ('PaperTitle', {
                'parent': self.styles['Title'],
                'fontSize': 20,
                'spaceAfter': 24,
                'spaceBefore': 12,
                'alignment': TA_CENTER,
                'textColor': darkblue,
                'fontName': 'Helvetica-Bold'
            }),
            ('AuthorStyle', {
                'parent': self.styles['Normal'],
                'fontSize': 12,
                'spaceAfter': 6,
                'alignment': TA_CENTER,
                'textColor': darkgreen
            }),
            ('AbstractTitle', {
                'parent': self.styles['Heading2'],
                'fontSize': 14,
                'spaceAfter': 8,
                'spaceBefore': 20,
                'alignment': TA_CENTER,
                'textColor': darkblue,
                'fontName': 'Helvetica-Bold'
            }),
            ('AbstractText', {
                'parent': self.styles['Normal'],
                'fontSize': 10,
                'spaceAfter': 16,
                'spaceBefore': 8,
                'alignment': TA_JUSTIFY,
                'leftIndent': 30,
                'rightIndent': 30,
                'borderWidth': 1,
                'borderColor': colors.grey,
                'borderPadding': 10
            }),
            ('KeywordsStyle', {
                'parent': self.styles['Normal'],
                'fontSize': 10,
                'spaceAfter': 20,
                'alignment': TA_LEFT,
                'leftIndent': 30,
                'fontName': 'Helvetica-Bold'
            }),
            ('MainHeading', {
                'parent': self.styles['Heading1'],
                'fontSize': 14,
                'spaceAfter': 12,
                'spaceBefore': 20,
                'textColor': darkblue,
                'fontName': 'Helvetica-Bold'
            }),
            ('SubHeading', {
                'parent': self.styles['Heading3'],
                'fontSize': 12,
                'spaceAfter': 8,
                'spaceBefore': 12,
                'textColor': darkgreen,
                'fontName': 'Helvetica-Bold'
            }),
            ('ResearchBodyText', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'spaceAfter': 8,
                'alignment': TA_JUSTIFY,
                'leftIndent': 0,
                'firstLineIndent': 20
            }),
            ('CitationStyle', {
                'parent': self.styles['Normal'],
                'fontSize': 10,
                'spaceAfter': 4,
                'leftIndent': 30,
                'fontName': 'Helvetica',
                'alignment': TA_JUSTIFY
            }),
            ('FooterStyle', {
                'parent': self.styles['Normal'],
                'fontSize': 8,
                'textColor': colors.grey,
                'alignment': TA_CENTER
            }),
            ('TOCHeading', {
                'parent': self.styles['Heading2'],
                'fontSize': 16,
                'spaceAfter': 12,
                'alignment': TA_CENTER,
                'textColor': darkblue
            }),
            ('TOCEntry', {
                'parent': self.styles['Normal'],
                'fontSize': 11,
                'spaceAfter': 4,
                'leftIndent': 20
            })
        ]
        
        # Add styles only if they don't exist
        for style_name, style_props in custom_styles:
            if style_name not in self.styles:
                self.styles.add(ParagraphStyle(name=style_name, **style_props))
    
    def clean_text(self, text: str) -> str:
        """Clean text for PDF generation"""
        if not text:
            return ""
        
        text = str(text)
        # Escape XML/HTML entities
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_title_page(self, brief_data: Dict) -> List:
        """Create a professional title page"""
        story = []
        
        # Add some space from top
        story.append(Spacer(1, 80))
        
        # Title
        title = self.clean_text(brief_data.get('topic', 'Research Analysis'))
        if not title.endswith(('Analysis', 'Study', 'Research', 'Review', 'Report')):
            title += ": A Comprehensive Analysis"
        
        story.append(Paragraph(title, self.styles['PaperTitle']))
        story.append(Spacer(1, 40))
        
        # Author information
        username = brief_data.get('user_id', 'Anonymous Researcher')[:8] + "..."
        story.append(Paragraph(f"Research conducted by: User {username}", self.styles['AuthorStyle']))
        story.append(Spacer(1, 20))
        
        # Institution/Source
        story.append(Paragraph("Generated by Context-Aware Research System", self.styles['AuthorStyle']))
        story.append(Spacer(1, 60))
        
        # Date
        date_str = datetime.now().strftime('%B %d, %Y')
        story.append(Paragraph(date_str, self.styles['AuthorStyle']))
        
        # Research details table
        story.append(Spacer(1, 80))
        
        details_data = [
            ['Research Parameters', ''],
            ['Topic:', self.clean_text(brief_data.get('topic', 'N/A'))],
            ['Target Audience:', self.clean_text(brief_data.get('audience', 'General'))],
            ['Research Depth:', f"Level {brief_data.get('depth', 'N/A')} of 5"],
            ['Research Type:', 'Follow-up Study' if brief_data.get('is_follow_up') else 'Initial Study'],
            ['Sources Analyzed:', str(len(brief_data.get('references', [])))],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        table = Table(details_data, colWidths=[2.5*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), HexColor('#4472C4')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (0, -1), HexColor('#F8F9FA')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(table)
        story.append(PageBreak())
        
        return story
    
    def create_table_of_contents(self, brief_data: Dict) -> List:
        """Create table of contents"""
        story = []
        
        story.append(Paragraph("Table of Contents", self.styles['TOCHeading']))
        story.append(Spacer(1, 20))
        
        toc_items = [
            "Abstract",
            "1. Introduction", 
            "2. Literature Review and Background",
            "3. Methodology",
            "4. Analysis and Findings",
            "5. Discussion",
            "6. Conclusions and Recommendations",
            "7. Future Research Directions",
            "References"
        ]
        
        # Add section-specific TOC items
        sections = brief_data.get('sections', [])
        if len(sections) > 4:
            # If we have many sections, add them as subsections
            for i, section in enumerate(sections[4:8], 1):  # Show first 4 as subsections
                heading = section.get('heading', f'Section {i+4}')
                toc_items.insert(-2, f"    4.{i} {self.clean_text(heading)[:50]}")
        
        for item in toc_items:
            story.append(Paragraph(item, self.styles['TOCEntry']))
            story.append(Spacer(1, 2))
        
        story.append(PageBreak())
        return story
    
    def create_abstract(self, brief_data: Dict) -> List:
        """Create abstract section"""
        story = []
        
        story.append(Paragraph("Abstract", self.styles['AbstractTitle']))
        
        # Generate abstract from thesis and first sections
        thesis = brief_data.get('thesis', '')
        sections = brief_data.get('sections', [])
        
        abstract_text = ""
        if thesis:
            abstract_text = f"This research examines {brief_data.get('topic', 'the specified topic')} with a focus on {audience_text[brief_data.get('audience', 'general')]}. {thesis} "
        
        # Add summary from first few sections
        if sections:
            for section in sections[:2]:  # Use first 2 sections for abstract
                content = section.get('content', '')
                if content:
                    # Extract first sentence or first 100 characters
                    first_sentence = content.split('.')[0][:150] + '...' if len(content) > 150 else content.split('.')[0]
                    abstract_text += f"{first_sentence}. "
        
        # Add methodology note
        if len(brief_data.get('references', [])) > 0:
            ref_count = len(brief_data.get('references', []))
            abstract_text += f"This analysis synthesizes findings from {ref_count} sources to provide comprehensive insights into the research topic."
        
        if not abstract_text:
            abstract_text = f"This research provides a comprehensive analysis of {brief_data.get('topic', 'the specified research topic')} through systematic examination of available literature and sources. The study aims to provide insights for {brief_data.get('audience', 'general audiences')} and contributes to the understanding of key concepts, challenges, and opportunities in this domain."
        
        story.append(Paragraph(self.clean_text(abstract_text), self.styles['AbstractText']))
        
        # Keywords
        topic_words = brief_data.get('topic', '').split()
        keywords = topic_words[:5] if len(topic_words) >= 3 else ['research', 'analysis', 'study']
        story.append(Paragraph(f"<b>Keywords:</b> {', '.join(keywords)}", self.styles['KeywordsStyle']))
        
        story.append(Spacer(1, 20))
        return story
    
    def create_full_research_paper(self, brief_data: Dict) -> List:
        """Create a complete research paper structure"""
        story = []
        
        # 1. Introduction
        story.append(Paragraph("1. Introduction", self.styles['MainHeading']))
        
        intro_text = f"""The field of {brief_data.get('topic', 'research')} has gained significant attention in recent years due to its implications for {brief_data.get('audience', 'various stakeholders')}. This research aims to provide a comprehensive analysis of key aspects, challenges, and opportunities within this domain.

The primary objective of this study is to examine {brief_data.get('topic', 'the research topic')} through a systematic review of available literature and evidence. This research is designed to serve {brief_data.get('audience', 'general')} audiences by providing accessible yet thorough insights into the subject matter."""
        
        if brief_data.get('thesis'):
            intro_text += f"\n\nThe central thesis of this research posits that {self.clean_text(brief_data['thesis'])}"
        
        story.append(Paragraph(self.clean_text(intro_text), self.styles['ResearchBodyText']))
        story.append(Spacer(1, 16))
        
        # 2. Literature Review and Background
        story.append(Paragraph("2. Literature Review and Background", self.styles['MainHeading']))
        
        lit_review = f"""The existing literature on {brief_data.get('topic', 'this topic')} reveals a complex landscape of research findings, theoretical frameworks, and practical applications. This section synthesizes key contributions from the field and identifies gaps that this research aims to address.

Previous studies have established foundational understanding in several key areas. The evolution of research in this field shows progression from early theoretical models to more sophisticated empirical investigations."""
        
        # Add content from first few sections if available
        sections = brief_data.get('sections', [])
        if len(sections) > 0 and sections[0].get('content'):
            lit_review += f"\n\n{self.clean_text(sections[0].get('content', ''))}"
        
        story.append(Paragraph(self.clean_text(lit_review), self.styles['ResearchBodyText']))
        story.append(Spacer(1, 16))
        
        # 3. Methodology
        story.append(Paragraph("3. Methodology", self.styles['MainHeading']))
        
        methodology = f"""This research employs a systematic approach to analyze and synthesize information related to {brief_data.get('topic', 'the research topic')}. The methodology consists of several key components:

<b>3.1 Research Design:</b> This study utilizes a comprehensive analytical framework designed to examine multiple dimensions of the research topic. The approach integrates both theoretical analysis and practical considerations relevant to {brief_data.get('audience', 'the target audience')}.

<b>3.2 Data Sources:</b> The research draws upon {len(brief_data.get('references', []))} primary sources, including academic literature, industry reports, and expert analyses. Sources were selected based on relevance, credibility, and recency.

<b>3.3 Analysis Framework:</b> The analysis employs a multi-level approach, examining both macro-level trends and micro-level details. This framework allows for comprehensive understanding while maintaining focus on practical implications."""
        
        story.append(Paragraph(methodology, self.styles['ResearchBodyText']))
        story.append(Spacer(1, 16))
        
        # 4. Analysis and Findings
        story.append(Paragraph("4. Analysis and Findings", self.styles['MainHeading']))
        
        # Use research sections for findings
        findings_intro = "The analysis reveals several key findings that contribute to our understanding of the research topic. These findings are organized thematically to provide clear insights into different aspects of the subject matter."
        story.append(Paragraph(findings_intro, self.styles['ResearchBodyText']))
        story.append(Spacer(1, 12))
        
        # Add sections as subsections of findings
        for i, section in enumerate(sections[1:] if len(sections) > 1 else sections, 1):
            heading = section.get('heading', f'Finding {i}')
            content = section.get('content', '')
            
            if heading and content:
                story.append(Paragraph(f"4.{i} {self.clean_text(heading)}", self.styles['SubHeading']))
                story.append(Paragraph(self.clean_text(content), self.styles['ResearchBodyText']))
                story.append(Spacer(1, 10))
        
        # 5. Discussion
        story.append(Paragraph("5. Discussion", self.styles['MainHeading']))
        
        discussion = f"""The findings presented in this research contribute to our understanding of {brief_data.get('topic', 'the research area')} in several important ways. The analysis reveals both opportunities and challenges that warrant careful consideration by {brief_data.get('audience', 'stakeholders')}.

The implications of these findings extend beyond the immediate scope of this research. They suggest directions for future investigation and highlight areas where additional attention may be needed. The research also identifies potential applications and practical considerations for implementation."""
        
        # Add thesis implications if available
        if brief_data.get('thesis'):
            discussion += f"\n\nThe central thesis of this research - {self.clean_text(brief_data['thesis'])} - finds support through the analysis presented. This suggests that the theoretical framework underlying this research provides a solid foundation for understanding the key dynamics at play."
        
        story.append(Paragraph(self.clean_text(discussion), self.styles['ResearchBodyText']))
        story.append(Spacer(1, 16))
        
        # 6. Conclusions and Recommendations
        story.append(Paragraph("6. Conclusions and Recommendations", self.styles['MainHeading']))
        
        conclusions = f"""This research has examined {brief_data.get('topic', 'the research topic')} through a systematic analysis of available evidence and literature. Several key conclusions emerge from this investigation:

<b>Primary Conclusions:</b>
• The research topic represents a significant area of interest with implications for {brief_data.get('audience', 'multiple stakeholders')}
• Current understanding benefits from multi-dimensional analysis that considers various perspectives and approaches
• There are opportunities for further development and application of insights gained through this research

<b>Recommendations:</b>
• Continued research is needed to address remaining questions and gaps identified in this analysis
• Stakeholders should consider the practical implications of findings when making related decisions
• Future studies should build upon the foundation established by this research while exploring new dimensions"""
        
        story.append(Paragraph(conclusions, self.styles['ResearchBodyText']))
        story.append(Spacer(1, 16))
        
        # 7. Future Research Directions
        story.append(Paragraph("7. Future Research Directions", self.styles['MainHeading']))
        
        future_research = f"""This research identifies several areas where future investigation could provide additional value:

<b>Methodological Extensions:</b> Future studies could employ alternative methodological approaches to validate and extend the findings presented here. Longitudinal studies, comparative analyses, and empirical investigations could provide additional insights.

<b>Scope Expansion:</b> While this research provides a solid foundation, there are opportunities to expand the scope to include additional dimensions, populations, or contexts relevant to {brief_data.get('topic', 'the research area')}.

<b>Practical Applications:</b> Additional research focused on implementation and practical application of findings could bridge the gap between theoretical understanding and real-world application."""
        
        story.append(Paragraph(future_research, self.styles['ResearchBodyText']))
        story.append(Spacer(1, 20))
        
        return story
    
    def create_references_section(self, brief_data: Dict) -> List:
        """Create properly formatted references section"""
        story = []
        
        references = brief_data.get('references', [])
        if not references:
            return story
        
        story.append(Paragraph("References", self.styles['MainHeading']))
        story.append(Spacer(1, 12))
        
        # Sort references alphabetically by title
        sorted_refs = sorted(references, key=lambda x: x.get('title', '').lower())
        
        for i, ref in enumerate(sorted_refs, 1):
            title = self.clean_text(ref.get('title', f'Source {i}'))
            url = ref.get('url', '')
            
            # Format reference in academic style
            if url:
                # Try to extract domain for source identification
                try:
                    domain = urlparse(url).netloc.replace('www.', '')
                    ref_text = f"[{i}] {title}. Retrieved from {domain}. Available at: {url}"
                except:
                    ref_text = f"[{i}] {title}. Available at: {url}"
            else:
                ref_text = f"[{i}] {title}."
            
            story.append(Paragraph(ref_text, self.styles['CitationStyle']))
            story.append(Spacer(1, 6))
        
        return story
    
    def add_page_numbers(self, canvas, doc):
        """Add page numbers and footer to each page"""
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(letter[0] - 50, 30, text)
        
        # Add footer with generation info
        footer_text = f"Generated by Research AI System - {datetime.now().strftime('%Y-%m-%d')}"
        canvas.drawCentredString(letter[0]/2, 15, footer_text)
        canvas.restoreState()

# Global dictionary for audience descriptions
audience_text = {
    'general': 'general audiences seeking accessible information',
    'academic': 'academic researchers and scholars',
    'business': 'business professionals and decision-makers', 
    'technical': 'technical experts and practitioners',
    'student': 'students and educational institutions',
    'policy': 'policy makers and governmental organizations'
}

def generate_pdf_report(brief_data: Dict, output_path: str, paper_format: str = "full") -> str:
    """
    Generate PDF report from research brief data
    
    Args:
        brief_data: Dictionary containing research brief data
        output_path: Path where PDF should be saved
        paper_format: "brief" for summary report or "full" for complete research paper
    
    Returns:
        Path to generated PDF file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    generator = ResearchPDFGenerator()
    story = []
    
    if paper_format == "full":
        # Generate complete research paper
        story.extend(generator.create_title_page(brief_data))
        story.extend(generator.create_table_of_contents(brief_data))
        story.extend(generator.create_abstract(brief_data))
        story.append(PageBreak())
        story.extend(generator.create_full_research_paper(brief_data))
        story.extend(generator.create_references_section(brief_data))
    else:
        # Generate brief summary report (original functionality)
        story.extend(generator.create_title_page(brief_data))
        
        # Add thesis if available
        thesis = brief_data.get('thesis')
        if thesis:
            story.append(Paragraph("Executive Summary", generator.styles['MainHeading']))
            story.append(Paragraph(generator.clean_text(thesis), generator.styles['ResearchBodyText']))
            story.append(Spacer(1, 16))
        
        # Add sections
        sections = brief_data.get('sections', [])
        if sections:
            story.append(Paragraph("Key Findings", generator.styles['MainHeading']))
            for i, section in enumerate(sections, 1):
                heading = section.get('heading', f'Finding {i}')
                content = section.get('content', '')
                if heading and content:
                    story.append(Paragraph(f"{i}. {generator.clean_text(heading)}", generator.styles['SubHeading']))
                    story.append(Paragraph(generator.clean_text(content), generator.styles['ResearchBodyText']))
                    story.append(Spacer(1, 10))
        
        story.extend(generator.create_references_section(brief_data))
    
    # Build PDF
    try:
        doc.build(story, onFirstPage=generator.add_page_numbers, onLaterPages=generator.add_page_numbers)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to generate PDF: {str(e)}")

def generate_research_paper(brief_data: Dict, output_path: str) -> str:
    """Generate a full research paper PDF"""
    return generate_pdf_report(brief_data, output_path, "full")

def generate_brief_summary(brief_data: Dict, output_path: str) -> str:
    """Generate a brief summary PDF"""
    return generate_pdf_report(brief_data, output_path, "brief")

if __name__ == "__main__":
    # Test the PDF generator
    test_data = {
        "topic": "Artificial Intelligence in Healthcare",
        "thesis": "AI has the potential to revolutionize healthcare delivery while raising important ethical considerations.",
        "audience": "healthcare",
        "depth": 4,
        "is_follow_up": False,
        "sections": [
            {
                "heading": "Current Applications",
                "content": "AI is currently being used in diagnostic imaging, drug discovery, and patient monitoring systems."
            },
            {
                "heading": "Benefits and Opportunities", 
                "content": "The technology offers improved accuracy, reduced costs, and enhanced patient outcomes."
            }
        ],
        "references": [
            {
                "title": "AI in Healthcare: A Comprehensive Review",
                "url": "https://example.com/ai-healthcare"
            }
        ],
        "session_id": "test-123",
        "user_id": "test-user"
    }
    
    # Generate both formats
    generate_research_paper(test_data, "test_research_paper.pdf")
    generate_brief_summary(test_data, "test_brief_summary.pdf")
    print("Test PDFs generated successfully!")