import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # 16:9 Widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors (Dark Forest / Earthy Landslide theme)
    BG_COLOR = RGBColor(24, 36, 30)         # Deep forest slate
    CARD_BG = RGBColor(34, 50, 42)          # Elevated card surface
    CARD_BORDER = RGBColor(200, 215, 205)   # Crisp light border
    ACCENT_GREEN = RGBColor(94, 194, 183)   # Mint accent
    ACCENT_GOLD = RGBColor(220, 175, 105)   # Amber highlight
    TEXT_WHITE = RGBColor(255, 255, 255)    # Primary text
    TEXT_MUTED = RGBColor(170, 185, 175)    # Secondary muted text
    PILL_BG = RGBColor(255, 255, 255)       # White badge/pill
    PILL_TEXT = RGBColor(24, 36, 30)        # Dark badge text

    FONT_TITLE = "Trebuchet MS"
    FONT_BODY = "Segoe UI"

    def set_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_pill="CMSC 170 FINAL PROJECT"):
        # Category pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.45), Inches(3.2), Inches(0.42))
        pill.fill.solid()
        pill.fill.fore_color.rgb = PILL_BG
        pill.line.fill.background()
        tf = pill.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = category_pill
        run.font.name = FONT_TITLE
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = PILL_TEXT

        # Title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.9))
        tf_t = tb.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        run_t = p_t.add_run()
        run_t.text = title_text.upper()
        run_t.font.name = FONT_TITLE
        run_t.font.size = Pt(28)
        run_t.font.bold = True
        run_t.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1)

    # Main Card
    main_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.1), Inches(10.933), Inches(5.3))
    main_card.fill.solid()
    main_card.fill.fore_color.rgb = CARD_BG
    main_card.line.color.rgb = CARD_BORDER
    main_card.line.width = Pt(1.5)

    # Top Pill
    t_pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.66), Inches(1.6), Inches(4.0), Inches(0.5))
    t_pill.fill.solid()
    t_pill.fill.fore_color.rgb = PILL_BG
    t_pill.line.fill.background()
    p = t_pill.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "VON DOOMSDAY ON DEC 18"
    r.font.name = FONT_TITLE
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    # Main Title
    tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(2.35), Inches(10.33), Inches(1.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = "EARLY SEVERITY ASSESSMENT OF LANDSLIDE EVENTS"
    r1.font.name = FONT_TITLE
    r1.font.size = Pt(32)
    r1.font.bold = True
    r1.font.color.rgb = TEXT_WHITE

    p2 = tf1.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "A Comparison of Naive Bayes, K-Nearest Neighbors, and Decision Trees from Scratch"
    r2.font.name = FONT_BODY
    r2.font.size = Pt(16)
    r2.font.color.rgb = ACCENT_GREEN

    # Presenters Line
    tb_pres = s1.shapes.add_textbox(Inches(1.5), Inches(4.5), Inches(10.33), Inches(1.2))
    tf_p = tb_pres.text_frame
    p_pr = tf_p.paragraphs[0]
    p_pr.alignment = PP_ALIGN.CENTER
    r_pr = p_pr.add_run()
    r_pr.text = "[1] Abonal, Julian   ·   [2] Madelo, Allen   ·   [3] Manuel, Ivan Erick   ·   [4] Noble, Alexus Benhur"
    r_pr.font.name = FONT_BODY
    r_pr.font.size = Pt(13)
    r_pr.font.bold = True
    r_pr.font.color.rgb = TEXT_WHITE

    p_cr = tf_p.add_paragraph()
    p_cr.alignment = PP_ALIGN.CENTER
    r_cr = p_cr.add_run()
    r_cr.text = "CMSC 170: Introduction to Artificial Intelligence · 1st Semester AY 2026–2027"
    r_cr.font.name = FONT_BODY
    r_cr.font.size = Pt(12)
    r_cr.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------------------
    # SLIDE 2: Problem & Motivation
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2)
    add_header(s2, "Problem Statement & Operational Motivation", "THE CHALLENGE")

    # Card 1: The Disaster Reality
    c1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7))
    c1.fill.solid()
    c1.fill.fore_color.rgb = CARD_BG
    c1.line.color.rgb = CARD_BORDER
    c1.line.width = Pt(1.5)

    pill1 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(2.3), Inches(2.8), Inches(0.45))
    pill1.fill.solid()
    pill1.fill.fore_color.rgb = PILL_BG
    p = pill1.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "THE OPERATIONAL PROBLEM"
    r.font.name = FONT_TITLE
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    tb = s2.shapes.add_textbox(Inches(1.1), Inches(3.0), Inches(5.0), Inches(3.4))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets1 = [
        "Landslides strike rapidly, burying communities and blocking roads.",
        "Emergency triage bottleneck: Rescue teams receive dozens of reports simultaneously during extreme storms.",
        "Responders have limited personnel, excavators, and medical units."
    ]
    for i, b in enumerate(bullets1):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(14)
        r = p.add_run()
        r.text = "• " + b
        r.font.name = FONT_BODY
        r.font.size = Pt(14)
        r.font.color.rgb = TEXT_WHITE

    # Card 2: The AI Solution
    c2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(2.0), Inches(5.6), Inches(4.7))
    c2.fill.solid()
    c2.fill.fore_color.rgb = CARD_BG
    c2.line.color.rgb = CARD_BORDER
    c2.line.width = Pt(1.5)

    pill2 = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(2.3), Inches(2.8), Inches(0.45))
    pill2.fill.solid()
    pill2.fill.fore_color.rgb = PILL_BG
    p = pill2.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "THE AI TASK & GOAL"
    r.font.name = FONT_TITLE
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    tb2 = s2.shapes.add_textbox(Inches(7.2), Inches(3.0), Inches(5.0), Inches(3.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        "Binary Classification: Predict fatal = 1 (fatalities > 0) vs fatal = 0 immediately at report time.",
        "Decision Support System: Automatically flag high-risk landslide reports so first responders prioritize life-saving aid.",
        "Strict Requirement: 100% from-scratch implementation without Scikit-learn or Pandas."
    ]
    for i, b in enumerate(bullets2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_after = Pt(14)
        r = p.add_run()
        r.text = "• " + b
        r.font.name = FONT_BODY
        r.font.size = Pt(14)
        r.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 3: Dataset & Features
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3)
    add_header(s3, "NASA Global Landslide Catalog", "DATASET & PREPROCESSING")

    # 3 Stat Cards
    stat_configs = [
        ("6,000 SAMPLES", "Experimental Sample Size", "9,648 total events in catalog\nSampled 6,000 with complete fatality counts"),
        ("25.4% FATAL", "Natural Class Imbalance", "1,526 Fatal (Positive Class)\n4,474 Non-Fatal (Negative Class)"),
        ("9 FEATURES", "Predictor Attributes", "5 Categorical (One-hot encoded)\n4 Continuous (Standardized z-scores)")
    ]

    for i, (stat, subtitle, desc) in enumerate(stat_configs):
        x = Inches(0.8 + i * 4.0)
        card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.0), Inches(3.73), Inches(3.7))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        tb = s3.shapes.add_textbox(x + Inches(0.2), Inches(2.2), Inches(3.33), Inches(3.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        r1 = p1.add_run()
        r1.text = stat
        r1.font.name = FONT_TITLE
        r1.font.size = Pt(24)
        r1.font.bold = True
        r1.font.color.rgb = ACCENT_GREEN

        p2 = tf.add_paragraph()
        p2.space_after = Pt(12)
        r2 = p2.add_run()
        r2.text = subtitle.upper()
        r2.font.name = FONT_TITLE
        r2.font.size = Pt(11)
        r2.font.bold = True
        r2.font.color.rgb = TEXT_MUTED

        p3 = tf.add_paragraph()
        r3 = p3.add_run()
        r3.text = desc
        r3.font.name = FONT_BODY
        r3.font.size = Pt(13)
        r3.font.color.rgb = TEXT_WHITE

    # Bottom Banner for Features
    b_card = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0), Inches(11.73), Inches(0.85))
    b_card.fill.solid()
    b_card.fill.fore_color.rgb = CARD_BG
    b_card.line.color.rgb = CARD_BORDER
    b_card.line.width = Pt(1.2)

    tb_b = s3.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.33), Inches(0.75))
    tf_b = tb_b.text_frame
    p_b = tf_b.paragraphs[0]
    r_b1 = p_b.add_run()
    r_b1.text = "Features: "
    r_b1.font.bold = True
    r_b1.font.color.rgb = ACCENT_GOLD
    r_b2 = p_b.add_run()
    r_b2.text = "landslide_category, landslide_trigger, landslide_size, landslide_setting, location_accuracy | latitude, longitude, month, logpop"
    r_b2.font.color.rgb = TEXT_WHITE
    p_b.font.size = Pt(12)

    # -------------------------------------------------------------------------
    # SLIDE 4: The Three Models (Matches the Canva reference layout!)
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4)
    add_header(s4, "Three Machine Learning Methods from Scratch", "METHODOLOGY")

    models_info = [
        ("K-NEAREST NEIGHBORS", [
            "Euclidean distance in normalized feature space",
            "Inverse class-ratio voting to prevent fatal cases from being outvoted",
            "Tuned k values: 3, 5, 7, 11 (Best k=7 to 11)"
        ]),
        ("NAIVE BAYES", [
            "Quantile binning (10 bins) for continuous features",
            "Lidstone smoothing (alpha = 0.1) for zero-count values",
            "Probability cutoff threshold tuning: 0.3, 0.5, 0.7"
        ]),
        ("DECISION TREE", [
            "Recursive binary splitting optimizing Gini Impurity reduction",
            "Training data oversampling to enforce 50/50 class balance",
            "Tuned max depth: 3, 5, 7 (Best depth=7 for complex rules)"
        ])
    ]

    for i, (m_name, bullets) in enumerate(models_info):
        x = Inches(0.8 + i * 4.0)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.0), Inches(3.73), Inches(4.7))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        # White pill badge at the bottom of the card like in user's image
        pill = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(0.3), Inches(2.3), Inches(3.13), Inches(0.5))
        pill.fill.solid()
        pill.fill.fore_color.rgb = PILL_BG
        pill.line.fill.background()
        p = pill.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = m_name
        r.font.name = FONT_TITLE
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = PILL_TEXT

        # Bullet text inside card
        tb = s4.shapes.add_textbox(x + Inches(0.25), Inches(3.1), Inches(3.23), Inches(3.4))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.space_after = Pt(14)
            r = p.add_run()
            r.text = "• " + b
            r.font.name = FONT_BODY
            r.font.size = Pt(13)
            r.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 5: Experimental Setup & Asymmetric Cost
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5)
    add_header(s5, "Fair Evaluation & Asymmetric Cost Function", "EXPERIMENTAL SETUP")

    # Card 1: 3-Fold Cross-Validation
    c1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7))
    c1.fill.solid()
    c1.fill.fore_color.rgb = CARD_BG
    c1.line.color.rgb = CARD_BORDER
    c1.line.width = Pt(1.5)

    pill1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(2.3), Inches(2.8), Inches(0.45))
    pill1.fill.solid()
    pill1.fill.fore_color.rgb = PILL_BG
    p = pill1.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "3-FOLD CROSS-VALIDATION"
    r.font.name = FONT_TITLE
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    tb1 = s5.shapes.add_textbox(Inches(1.1), Inches(3.0), Inches(5.0), Inches(3.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    bullets_cv = [
        "Rigorous 3-Fold Cross-Validation on random shuffle (Seed 42).",
        "Each fold: 4,000 training samples / 2,000 held-out test samples.",
        "Hyperparameter Selection: Inner 80/20 train split selects best parameter based on validation cost, then retrained on full fold.",
        "Strict Leakage Prevention: Normalization and encoding statistics computed exclusively from training folds."
    ]
    for i, b in enumerate(bullets_cv):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.space_after = Pt(10)
        r = p.add_run()
        r.text = "• " + b
        r.font.name = FONT_BODY
        r.font.size = Pt(13)
        r.font.color.rgb = TEXT_WHITE

    # Card 2: Cost Function
    c2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(2.0), Inches(5.6), Inches(4.7))
    c2.fill.solid()
    c2.fill.fore_color.rgb = CARD_BG
    c2.line.color.rgb = CARD_BORDER
    c2.line.width = Pt(1.5)

    pill2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(2.3), Inches(3.0), Inches(0.45))
    pill2.fill.solid()
    pill2.fill.fore_color.rgb = PILL_BG
    p = pill2.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "THE ASYMMETRIC COST METRIC"
    r.font.name = FONT_TITLE
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    tb2 = s5.shapes.add_textbox(Inches(7.2), Inches(3.0), Inches(5.0), Inches(3.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    r = p.add_run()
    r.text = "Cost = (1 × False Positives) + (2 × False Negatives)"
    r.font.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = ACCENT_GOLD
    p.space_after = Pt(14)

    bullets_cost = [
        "Accuracy alone is misleading: A naive model that always predicts 'Non-Fatal' gets 74.57% accuracy but saves zero lives.",
        "False Positive (Cost = 1): A false alarm wastes inspection time and fuel.",
        "False Negative (Cost = 2): A missed fatal event leaves victims without rescue teams.",
        "Goal: Minimize Total Cost and Maximize Recall."
    ]
    for b in bullets_cost:
        p = tf2.add_paragraph()
        p.space_after = Pt(8)
        r = p.add_run()
        r.text = "• " + b
        r.font.name = FONT_BODY
        r.font.size = Pt(13)
        r.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 6: Performance Benchmark Table
    # -------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6)
    add_header(s6, "Comprehensive 3-Fold Benchmark Results", "RESULTS")

    # Add Table
    rows, cols = 5, 6
    left, top, width, height = Inches(0.8), Inches(2.0), Inches(11.73), Inches(3.6)
    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table_data = [
        ["Method", "Accuracy", "Precision", "Recall", "F1-Score", "Cost (FP + 2*FN)"],
        ["Always Non-Fatal (Baseline)", "74.57%", "0.00%", "0.00%", "0.00%", "3,052"],
        ["Naive Bayes (Quantiles)", "74.02%", "49.43%", "85.66%", "62.63%", "1,779"],
        ["k-Nearest Neighbors (KNN)", "75.50% (Best)", "51.35% (Best)", "79.61%", "62.22%", "1,782"],
        ["Decision Tree (Balanced)", "73.95%", "49.36%", "91.11% (Best)", "64.02% (Best)", "1,699 (LOWEST)"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            run = p.runs[0]
            run.font.name = FONT_BODY
            run.font.size = Pt(13)
            if r_idx == 0:
                run.font.bold = True
                run.font.color.rgb = TEXT_WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(16, 26, 20)
            elif r_idx == 4:
                run.font.bold = True
                run.font.color.rgb = ACCENT_GOLD
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(40, 60, 50)
            else:
                run.font.color.rgb = TEXT_WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = CARD_BG

    # Bottom Callout Card
    call_card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.9), Inches(11.73), Inches(1.0))
    call_card.fill.solid()
    call_card.fill.fore_color.rgb = CARD_BG
    call_card.line.color.rgb = ACCENT_GREEN
    call_card.line.width = Pt(1.5)

    tb_c = s6.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.33), Inches(0.9))
    tf_c = tb_c.text_frame
    p_c = tf_c.paragraphs[0]
    r_c1 = p_c.add_run()
    r_c1.text = "KEY TAKEAWAY: "
    r_c1.font.bold = True
    r_c1.font.color.rgb = ACCENT_GREEN
    r_c2 = p_c.add_run()
    r_c2.text = "Decision Tree wins on operational cost (1,699) by achieving 91.11% Recall. KNN achieved the highest raw accuracy (75.50%) and precision (51.35%)."
    r_c2.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 7: Confusion Matrix & Detection Counts
    # -------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7)
    add_header(s7, "Detection Counts & Confusion Matrix (6,000 Events)", "DETECTION BREAKDOWN")

    counts = [
        ("DEADLY EVENTS MISSED", [
            ("Decision Tree:", "Only 136 Missed (Best)"),
            ("Naive Bayes:", "220 Missed"),
            ("k-NN:", "312 Missed"),
            ("Baseline:", "1,526 Missed (Catastrophic)")
        ]),
        ("FATAL EVENTS CAUGHT", [
            ("Decision Tree:", "1,390 Caught (91.11%)"),
            ("Naive Bayes:", "1,306 Caught (85.66%)"),
            ("k-NN:", "1,214 Caught (79.61%)"),
            ("Baseline:", "0 Caught (0.00%)")
        ]),
        ("FALSE ALARM BURDEN", [
            ("Decision Tree:", "1,427 False Alarms"),
            ("Naive Bayes:", "1,339 False Alarms"),
            ("k-NN:", "1,158 False Alarms (Lowest)"),
            ("Baseline:", "0 False Alarms")
        ])
    ]

    for i, (title, items) in enumerate(counts):
        x = Inches(0.8 + i * 4.0)
        card = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.0), Inches(3.73), Inches(4.7))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        pill = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(0.3), Inches(2.3), Inches(3.13), Inches(0.45))
        pill.fill.solid()
        pill.fill.fore_color.rgb = PILL_BG
        p = pill.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = title
        r.font.name = FONT_TITLE
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = PILL_TEXT

        tb = s7.shapes.add_textbox(x + Inches(0.25), Inches(3.1), Inches(3.23), Inches(3.4))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, (lbl, val) in enumerate(items):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.space_after = Pt(12)
            r1 = p.add_run()
            r1.text = lbl + " "
            r1.font.bold = True
            r1.font.color.rgb = ACCENT_GOLD if "Best" in val or "91.11%" in val else TEXT_WHITE
            r2 = p.add_run()
            r2.text = val
            r2.font.color.rgb = ACCENT_GREEN if "Best" in val or "91.11%" in val else TEXT_MUTED

    # -------------------------------------------------------------------------
    # SLIDE 8: Visual Comparison (Embeds method_comparison.png)
    # -------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8)
    add_header(s8, "Visualizing Metric Trade-offs Across Models", "GRAPHICAL BENCHMARK")

    # Embed plot if it exists
    img_path = "method_comparison.png"
    if os.path.exists(img_path):
        s8.shapes.add_picture(img_path, Inches(0.8), Inches(2.0), width=Inches(7.2))
    
    # Text card beside the image
    card_side = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(2.0), Inches(4.2), Inches(4.7))
    card_side.fill.solid()
    card_side.fill.fore_color.rgb = CARD_BG
    card_side.line.color.rgb = CARD_BORDER
    card_side.line.width = Pt(1.5)

    tb_s = s8.shapes.add_textbox(Inches(8.5), Inches(2.2), Inches(3.8), Inches(4.3))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True
    takeaways = [
        ("The Accuracy Illusion:", "Baseline gets 74.57% accuracy simply by ignoring fatal landslides. True model utility is demonstrated by Recall and Cost."),
        ("The Precision Ceiling:", "All models settle at ~49% to 51% precision. These 9 metadata features cannot completely eliminate false alarms."),
        ("Operational Triage:", "In life-critical disasters, catching 91.11% of casualties far outweighs responding to false alerts.")
    ]
    for i, (hd, txt) in enumerate(takeaways):
        p = tf_s.paragraphs[0] if i == 0 else tf_s.add_paragraph()
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = hd + "\n"
        r1.font.bold = True
        r1.font.size = Pt(13)
        r1.font.color.rgb = ACCENT_GREEN
        r2 = p.add_run()
        r2.text = txt
        r2.font.size = Pt(12)
        r2.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 9: Why Decision Trees Won (Discussion)
    # -------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s9)
    add_header(s9, "Why Decision Trees Outperformed Other Methods", "DISCUSSION")

    reasons = [
        ("FEATURE INTERACTIONS", [
            "Landslide danger is inherently non-linear.",
            "Example: Large landslide + uninhabited mountain = Non-fatal.",
            "Large landslide + high population (logpop) = High fatality.",
            "Decision trees capture these conjunctions; Naive Bayes assumes independent attributes."
        ]),
        ("OVERCOMING IMBALANCE", [
            "75% majority non-fatal class causes standard models to predict majority.",
            "Balanced training oversampling forced tree leaves to evaluate fatal signals equally.",
            "Depth 7 allowed sufficient branching to distinguish rare lethal conditions."
        ]),
        ("TRANSPARENCY & AUDITING", [
            "Emergency operators reject black-box models.",
            "Trees generate explicit if-else rules that civil defense officials can inspect and validate."
        ])
    ]

    for i, (title, bullets) in enumerate(reasons):
        x = Inches(0.8 + i * 4.0)
        card = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.0), Inches(3.73), Inches(4.7))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        pill = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(0.3), Inches(2.3), Inches(3.13), Inches(0.45))
        pill.fill.solid()
        pill.fill.fore_color.rgb = PILL_BG
        p = pill.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = title
        r.font.name = FONT_TITLE
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = PILL_TEXT

        tb = s9.shapes.add_textbox(x + Inches(0.25), Inches(3.0), Inches(3.23), Inches(3.5))
        tf = tb.text_frame
        tf.word_wrap = True
        for j, b in enumerate(bullets):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            p.space_after = Pt(10)
            r = p.add_run()
            r.text = "• " + b
            r.font.name = FONT_BODY
            r.font.size = Pt(12)
            r.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 10: Limitations & Future Work
    # -------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s10)
    add_header(s10, "Current Limitations & Future Enhancements", "LOOKING FORWARD")

    c1 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7))
    c1.fill.solid()
    c1.fill.fore_color.rgb = CARD_BG
    c1.line.color.rgb = CARD_BORDER
    c1.line.width = Pt(1.5)

    pill1 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(2.3), Inches(2.8), Inches(0.45))
    pill1.fill.solid()
    pill1.fill.fore_color.rgb = PILL_BG
    p = pill1.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "PROJECT LIMITATIONS"
    r.font.name = FONT_TITLE
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    tb1 = s10.shapes.add_textbox(Inches(1.1), Inches(3.0), Inches(5.0), Inches(3.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    limits = [
        "Random CV Splitting: Events from the exact same severe weather system can fall across train and test sets.",
        "Catalog Sample Size: Evaluated on 6,000 sampled events rather than full 9,648 dataset.",
        "Cost Weight Assumptions: FP=1 and FN=2 were chosen assumptions; real-world costs vary by municipal rescue capacity."
    ]
    for b in limits:
        p = tf1.add_paragraph()
        p.space_after = Pt(12)
        r = p.add_run()
        r.text = "• " + b
        r.font.size = Pt(13)
        r.font.color.rgb = TEXT_WHITE

    c2 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(2.0), Inches(5.6), Inches(4.7))
    c2.fill.solid()
    c2.fill.fore_color.rgb = CARD_BG
    c2.line.color.rgb = CARD_BORDER
    c2.line.width = Pt(1.5)

    pill2 = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.3), Inches(2.3), Inches(2.8), Inches(0.45))
    pill2.fill.solid()
    pill2.fill.fore_color.rgb = PILL_BG
    p = pill2.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "FUTURE IMPROVEMENTS"
    r.font.name = FONT_TITLE
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = PILL_TEXT

    tb2 = s10.shapes.add_textbox(Inches(7.2), Inches(3.0), Inches(5.0), Inches(3.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    futures = [
        "Time-based Splitting: Split by storm date or season to test true chronological forecasting.",
        "Full Catalog Training: Expand to all 9,648 events and include weather satellite radar rainfall feeds.",
        "Adaptive Decision Thresholds: Calibrate cost weights dynamically based on local emergency resource availability."
    ]
    for b in futures:
        p = tf2.add_paragraph()
        p.space_after = Pt(12)
        r = p.add_run()
        r.text = "• " + b
        r.font.size = Pt(13)
        r.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 11: Conclusion & Key Takeaways
    # -------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s11)
    add_header(s11, "Conclusions & Core Findings", "FINAL VERDICT")

    c_card = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(2.0), Inches(10.933), Inches(4.7))
    c_card.fill.solid()
    c_card.fill.fore_color.rgb = CARD_BG
    c_card.line.color.rgb = CARD_BORDER
    c_card.line.width = Pt(1.5)

    tb_end = s11.shapes.add_textbox(Inches(1.6), Inches(2.4), Inches(10.13), Inches(4.0))
    tf_end = tb_end.text_frame
    tf_end.word_wrap = True

    pts = [
        ("1. Decision Tree is the Optimal Triage Method:", "By catching 91.11% of all deadly events (1,390 out of 1,526) and missing only 136, Decision Tree achieved the lowest operational cost (1,699) and highest F1-score (64.02%)."),
        ("2. Real-world AI Evaluation Requires Domain-Specific Cost:", "High accuracy (74.57%) in imbalanced disaster classification is completely deceptive. Evaluating through an asymmetric cost function was critical to model utility."),
        ("3. From-Scratch Algorithms Are Viable:", "Implemented without Pandas or Scikit-learn, pure Python + NumPy models proved capable of robust cross-validated machine learning.")
    ]
    for i, (hd, txt) in enumerate(pts):
        p = tf_end.paragraphs[0] if i == 0 else tf_end.add_paragraph()
        p.space_after = Pt(16)
        r1 = p.add_run()
        r1.text = hd + " "
        r1.font.bold = True
        r1.font.size = Pt(15)
        r1.font.color.rgb = ACCENT_GOLD
        r2 = p.add_run()
        r2.text = txt
        r2.font.size = Pt(14)
        r2.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------------------
    # SLIDE 12: Q&A Slide
    # -------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s12)

    card12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.0), Inches(1.5), Inches(9.333), Inches(4.5))
    card12.fill.solid()
    card12.fill.fore_color.rgb = CARD_BG
    card12.line.color.rgb = CARD_BORDER
    card12.line.width = Pt(1.5)

    tb12 = s12.shapes.add_textbox(Inches(2.5), Inches(2.2), Inches(8.33), Inches(3.0))
    tf12 = tb12.text_frame
    p12 = tf12.paragraphs[0]
    p12.alignment = PP_ALIGN.CENTER
    r12 = p12.add_run()
    r12.text = "QUESTIONS & ANSWERS"
    r12.font.name = FONT_TITLE
    r12.font.size = Pt(36)
    r12.font.bold = True
    r12.font.color.rgb = TEXT_WHITE

    p12_2 = tf12.add_paragraph()
    p12_2.alignment = PP_ALIGN.CENTER
    p12_2.space_before = Pt(16)
    r12_2 = p12_2.add_run()
    r12_2.text = "Thank you! We are open for questions."
    r12_2.font.size = Pt(18)
    r12_2.font.color.rgb = ACCENT_GREEN

    p12_3 = tf12.add_paragraph()
    p12_3.alignment = PP_ALIGN.CENTER
    p12_3.space_before = Pt(20)
    r12_3 = p12_3.add_run()
    r12_3.text = "GitHub: github.com/KcireNavi/VON-DOOMSDAY-CMSC-170-PROJECT"
    r12_3.font.size = Pt(12)
    r12_3.font.color.rgb = TEXT_MUTED

    output_path = "presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    create_presentation()
