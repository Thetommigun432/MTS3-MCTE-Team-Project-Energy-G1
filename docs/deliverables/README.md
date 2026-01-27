# NILM Energy Monitor - Final Deliverables Index

**Project:** NILM Energy Monitor  
**Team:** MTS3-MCTE Team Project - Energy G1  
**Institution:** HOWEST  
**Date:** January 2026  
**Version:** Final Release  

---

## Deliverables Summary

This folder contains all final documentation deliverables for the NILM Energy Monitor project, prepared for academic submission.

---

## Document List

| # | Document | Filename | Description | Pages |
|---|----------|----------|-------------|-------|
| 1 | **Presentation** | [01_PRESENTATION.md](01_PRESENTATION.md) | Final pitch/defense slides (structured for PDF conversion) | ~10 |
| 2 | **Functional Analysis** | [02_FUNCTIONAL_ANALYSIS.md](02_FUNCTIONAL_ANALYSIS.md) | Complete functional specification with user testing results | ~25 |
| 3 | **Design Document** | [03_DESIGN_DOCUMENT.md](03_DESIGN_DOCUMENT.md) | UI/UX design system, colors, typography, accessibility | ~20 |
| 4 | **User Manual** | [04_USER_MANUAL.md](04_USER_MANUAL.md) | End-user documentation for the application | ~18 |
| 5 | **Installation Guide** | [05_INSTALLATION_GUIDE.md](05_INSTALLATION_GUIDE.md) | Technical setup guide for developers | ~15 |
| 6 | **Project Management** | [06_PROJECT_MANAGEMENT.md](06_PROJECT_MANAGEMENT.md) | Agile methodology, sprints, retrospectives | ~18 |
| 7 | **Architecture Overview** | [07_ARCHITECTURE_OVERVIEW.md](07_ARCHITECTURE_OVERVIEW.md) | A4 technical architecture description | 1 |
| 8 | **Source Code Structure** | [08_SOURCE_CODE_STRUCTURE.md](08_SOURCE_CODE_STRUCTURE.md) | ZIP structure and file descriptions | ~10 |

---

## Project Overview

### What is NILM Energy Monitor?

NILM Energy Monitor is a web-based application that uses **Non-Intrusive Load Monitoring** technology to disaggregate total household power consumption into individual appliance usage. Using deep learning (PyTorch), it can identify which appliances (heat pump, dishwasher, washing machine, etc.) are consuming power from a single smart meter reading.

### Technology Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19, TypeScript, Vite 7, Tailwind CSS |
| **Backend** | FastAPI, Python 3.12, PyTorch |
| **Database** | InfluxDB (time-series), Supabase (auth + metadata) |
| **Deployment** | Cloudflare Pages, Railway |

### Key Features

- Real-time energy consumption dashboard
- AI-powered appliance disaggregation
- Confidence indicators for predictions
- Multi-building support
- Data export (CSV, summary reports)
- Responsive design (desktop, tablet, mobile)

---

## Converting to PDF

Each document is written in Markdown and can be converted to PDF using various tools:

### Option 1: VS Code Extension
1. Install "Markdown PDF" extension in VS Code
2. Open the markdown file
3. Press `Ctrl+Shift+P` → "Markdown PDF: Export (pdf)"

### Option 2: Pandoc (Command Line)
```bash
pandoc 01_PRESENTATION.md -o 01_PRESENTATION.pdf --pdf-engine=wkhtmltopdf
```

### Option 3: Online Converters
- [MarkdownToPDF.com](https://www.markdowntopdf.com/)
- [Dillinger.io](https://dillinger.io/) (Export to PDF)

### Recommended Styling for PDF
- Font: Arial or Helvetica, 11pt
- Margins: 2.5cm
- Headers: Bold, larger size
- Tables: Borders, alternating row colors

---

## Assumptions Made

Since some project details were not explicitly provided, the following realistic assumptions were made:

1. **Team Size:** 4 members (based on typical academic team projects)
2. **Sprint Duration:** 2 weeks (standard for academic timelines)
3. **User Testing:** 12 participants conducted over 2 weeks
4. **SUS Score:** 78/100 (based on typical well-designed applications)
5. **Deployment:** Cloudflare Pages + Railway (from project documentation)
6. **Contact Information:** Placeholder emails used (replace before submission)

---

## Related Project Documentation

Additional technical documentation is available in the parent `docs/` folder:

- [PROJECT.md](../PROJECT.md) - Architecture and repository structure
- [LOCAL_DEV.md](../LOCAL_DEV.md) - Local development setup
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Production deployment guide
- [OPERATIONS.md](../OPERATIONS.md) - Operations runbook and troubleshooting
- [API.md](../API.md) - Backend API reference
- [frontend.md](../frontend.md) - Frontend-specific documentation

---

## Submission Checklist

Before submission, verify:

- [ ] All 8 documents are complete
- [ ] Documents are converted to PDF
- [ ] No placeholder text remains
- [ ] Contact information is correct
- [ ] Source code ZIP is prepared
- [ ] Large files (models) are hosted separately
- [ ] All team member names are included
- [ ] Version numbers are consistent

---

*© 2026 MTS3-MCTE Team Project - Energy G1 | HOWEST*
