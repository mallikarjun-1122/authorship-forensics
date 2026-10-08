from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_report(authorship_data, ai_score, drift) -> bytes:
    """Generate a clean forensic PDF report with black, white, and light green accents."""
    if isinstance(authorship_data, dict):
        adv_score = authorship_data.get("advanced_score", 0.0)
        basic_score = authorship_data.get("basic_score", 0.0)
        xai = authorship_data.get("xai", {})
    else:
        adv_score = float(authorship_data)
        basic_score = float(authorship_data)
        xai = {}

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter

    # Color palette: Black, White, Light Green
    c_black = colors.HexColor("#0d1117")
    c_light_green = colors.HexColor("#22c55e")
    c_muted = colors.HexColor("#4b5563")

    # Header Bar (Black with Light Green accent bar)
    pdf.setFillColor(c_black)
    pdf.rect(0, height - 70, width, 70, fill=1, stroke=0)
    pdf.setFillColor(c_light_green)
    pdf.rect(0, height - 74, width, 4, fill=1, stroke=0)

    # Title text
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 42, "AI Writing Forensics Analysis Report")
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(c_light_green)
    pdf.drawString(50, height - 58, "CONFIDENTIAL & FORENSIC ATTRIBUTION REPORT")

    y = height - 105

    def draw_section_header(title):
        nonlocal y
        if y < 80:
            pdf.showPage()
            y = height - 60
        pdf.setFillColor(c_black)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, title)
        pdf.setStrokeColor(c_light_green)
        pdf.setLineWidth(1.5)
        pdf.line(50, y - 4, 250, y - 4)
        y -= 22

    def draw_line(text, font="Helvetica", size=9.5, color=colors.black, indent=50):
        nonlocal y
        if y < 50:
            pdf.showPage()
            y = height - 60
        pdf.setFillColor(color)
        pdf.setFont(font, size)
        pdf.drawString(indent, y, text)
        y -= 14

    # 1. Executive Summary
    draw_section_header("1. Executive Summary")
    draw_line(f"• Advanced N-Gram Stylometric Similarity: {adv_score}%", font="Helvetica-Bold", color=c_black)
    draw_line(f"• Basic Feature Vector Similarity: {basic_score}%", color=c_muted)
    draw_line(f"• AI Generic Phrase & Uniformity Score: {ai_score.get('score', 0)} / 100", font="Helvetica-Bold", color=c_black)
    draw_line(f"• Sentence Burstiness (Length Variance): {ai_score.get('burstiness', 0.0)} (Penalty: +{ai_score.get('burst_penalty', 0)})", color=c_muted)
    y -= 10

    # 2. Explainable Stylometry Findings
    draw_section_header("2. Stylistic Fingerprint & Pattern Analysis")
    shared = xai.get("shared", [])
    missing = xai.get("missing", [])
    if shared:
        draw_line(f"• Dominant Shared Style Markers: {', '.join(shared[:8])}", color=c_black)
    else:
        draw_line("• Dominant Shared Style Markers: None detected or sample too brief.", color=c_muted)
    if missing:
        draw_line(f"• Prominent Known-Author Markers Absent: {', '.join(missing[:8])}", color=c_muted)
    y -= 10

    # 3. Style Drift (Paragraph Analysis)
    draw_section_header("3. Intra-Document Style Drift")
    if not drift:
        draw_line("• No paragraph segmentation detected.", color=c_muted)
    else:
        for item in drift:
            sim_str = "N/A" if item.get("similarity") is None else f"{item['similarity']}%"
            flag = item.get("flag", "Normal")
            flag_col = c_black if "High" not in flag else colors.HexColor("#b91c1c")
            draw_line(
                f"• Paragraph {item.get('paragraph', 1)}: Similarity: {sim_str} | Status: {flag}",
                color=flag_col
            )
    y -= 10

    # 4. AI Indicator Breakdown
    draw_section_header("4. AI Language Flag Breakdown")
    indicators = ai_score.get("indicators", [])
    if indicators:
        draw_line(f"• Matched High-Frequency AI Indicator Terms ({len(indicators)} detected):", color=c_black)
        draw_line(f"   {', '.join(indicators[:12])}", font="Helvetica-Oblique", color=c_muted)
    else:
        draw_line("• No synthetic buzzword indicators matched from reference catalogue.", color=c_muted)

    y -= 15
    draw_line("Note: These scores represent stylistic and statistical indicators, not absolute proof.", font="Helvetica-Oblique", size=8, color=c_muted)

    pdf.save()
    output.seek(0)
    return output.getvalue()
