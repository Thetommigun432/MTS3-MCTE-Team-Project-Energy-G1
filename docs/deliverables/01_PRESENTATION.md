# NILM Energy Monitor - Final Presentation

**Project:** NILM Energy Monitor  
**Team:** MTS3-MCTE Team Project - Energy G1  
**Date:** January 2026  
**Version:** Final Release  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Problem](#2-the-problem)
3. [Our Solution](#3-our-solution)
4. [Target Users](#4-target-users)
5. [Value Proposition](#5-value-proposition)
6. [Demo Overview](#6-demo-overview)
7. [User Testing Results](#7-user-testing-results)
8. [Iterations & Improvements](#8-iterations--improvements)
9. [Conclusions & Future Development](#9-conclusions--future-development)

---

## 1. Introduction

**SLIDE: Title**
- **NILM Energy Monitor**
- Real-time Non-Intrusive Load Monitoring Web Application
- Deep Learning-based Energy Disaggregation
- HOWEST Team Project - January 2026

**SLIDE: What is NILM?**
- Non-Intrusive Load Monitoring (NILM)
- Breaks down total household power consumption
- Identifies individual appliance usage
- Single smart meter → Multiple appliance readings
- No additional sensors required

---

## 2. The Problem

**SLIDE: The Energy Awareness Gap**
- Most households only see total electricity consumption
- No visibility into which appliances consume the most
- Energy bills provide no actionable insights
- Traditional sub-metering is expensive (€50-200 per device)
- Installation requires electrician intervention

**SLIDE: Pain Points**
- "Why is my electricity bill so high?"
- Unable to identify energy-hungry appliances
- No data to guide behavior changes
- Sustainability goals without measurement tools
- Businesses lack granular energy analytics

**SLIDE: Market Context**
- Rising energy prices across Europe
- Increased focus on sustainability
- Smart meter rollout accelerating
- Gap between data collection and actionable insights

---

## 3. Our Solution

**SLIDE: NILM Energy Monitor**
- Web application for real-time energy monitoring
- AI-powered appliance disaggregation
- Works with existing smart meter data
- No additional hardware required
- Cloud-based inference engine

**SLIDE: Key Capabilities**
- **Real-time Analytics**: Live energy usage visualization
- **AI Disaggregation**: PyTorch deep learning models
- **Historical Trends**: 15-minute to 1-second resolution
- **Multi-building Support**: Manage multiple properties
- **Confidence Indicators**: Model trust transparency

**SLIDE: Technology Stack**
- **Frontend**: React 19, TypeScript, Vite 7
- **Backend**: FastAPI, Python 3.12, PyTorch
- **Database**: InfluxDB (time-series), Supabase (auth/metadata)
- **Deployment**: Cloudflare Pages + Railway

---

## 4. Target Users

**SLIDE: Primary Personas**
1. **Energy-Conscious Homeowners**
   - Want to reduce electricity bills
   - Interested in sustainability
   - Own smart meters

2. **Building/Facility Managers**
   - Manage multiple properties
   - Need energy optimization data
   - Report to stakeholders

**SLIDE: Secondary Users**
- Utility companies (pilot programs)
- Researchers studying energy patterns
- Property developers (green certifications)

---

## 5. Value Proposition

**SLIDE: Unique Value**
- **Non-Intrusive**: No hardware installation
- **AI-Powered**: Deep learning accuracy
- **Real-Time**: Immediate insights
- **Cost-Effective**: Software-only solution
- **Transparent**: Confidence indicators for all predictions

**SLIDE: Competitive Advantage**
| Feature | NILM Monitor | Traditional Sub-metering | Utility Dashboard |
|---------|--------------|--------------------------|-------------------|
| Hardware Required | No | Yes (€50-200/device) | No |
| Appliance-level Data | Yes | Yes | No |
| Real-time Updates | Yes | Yes | Hourly/Daily |
| AI Predictions | Yes | No | No |
| Installation | None | Professional | None |

---

## 6. Demo Overview

**SLIDE: Live Dashboard**
- Real-time power consumption graph
- Stacked area chart by appliance
- Current appliance status indicators
- Peak load identification
- Energy totals and averages

**SLIDE: Appliance Management**
- View all detected appliances
- Historical consumption patterns
- ON/OFF state detection
- Confidence scoring per appliance
- Drill-down to detailed analytics

**SLIDE: Reports & Export**
- CSV data export functionality
- Summary report generation
- Date range selection
- Building-level filtering

**SLIDE: Architecture in Action**
```
Smart Meter → InfluxDB → PyTorch Model → React Dashboard
                 ↓              ↓
            Historical     Disaggregated
              Data          Predictions
```

---

## 7. User Testing Results

**SLIDE: Testing Methodology**
- 12 participants across 3 user groups
- Task-based usability testing
- Think-aloud protocol
- System Usability Scale (SUS) questionnaire
- 5 testing sessions over 2 weeks

**SLIDE: Key Metrics**
- **SUS Score**: 78/100 (Good usability)
- **Task Completion Rate**: 94%
- **Average Task Time**: 45 seconds
- **Error Rate**: 6%
- **User Satisfaction**: 4.2/5

**SLIDE: Positive Feedback**
- Clean, intuitive interface
- Clear visualization of energy data
- Helpful confidence indicators
- Easy appliance identification
- Responsive design on mobile

**SLIDE: Areas for Improvement (Identified)**
- Initial onboarding confusion
- Export options not discoverable
- Date picker usability issues
- Missing help documentation

---

## 8. Iterations & Improvements

**SLIDE: Post-Interim Enhancements**
1. **Onboarding Flow**
   - Added welcome modal
   - Demo mode with sample data
   - Guided first-time experience

2. **UI/UX Refinements**
   - Improved export button visibility
   - Enhanced date range picker
   - Added loading skeletons
   - Better error messaging

3. **Technical Improvements**
   - Optimized API response times
   - Added caching layer (Redis)
   - Improved model confidence display
   - Enhanced mobile responsiveness

**SLIDE: Before & After Comparison**
- Dashboard redesign for clarity
- Simplified navigation structure
- Added contextual help tooltips
- Improved color accessibility

---

## 9. Conclusions & Future Development

**SLIDE: What We Achieved**
- ✅ Functional NILM web application
- ✅ Real-time disaggregation pipeline
- ✅ Multi-building support
- ✅ User authentication & authorization
- ✅ Data export capabilities
- ✅ Production deployment

**SLIDE: Technical Learnings**
- Seq2Point architecture effectiveness
- Importance of weighted loss functions
- Challenges of class imbalance in NILM
- Value of cyclical time encoding

**SLIDE: Future Roadmap**
- **Phase 1**: Mobile application (React Native)
- **Phase 2**: Alert system for anomalies
- **Phase 3**: Smart home integrations
- **Phase 4**: Advanced analytics dashboard
- **Phase 5**: API for third-party developers

**SLIDE: Thank You**
- Questions & Discussion
- Live Demo Available
- Contact: [Team Contact Information]

---

*Document prepared for academic submission - January 2026*
