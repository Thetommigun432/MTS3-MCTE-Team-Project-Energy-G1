# NILM Energy Monitor - Functional Analysis

**Document Version:** Final (Post-User Testing)  
**Project:** NILM Energy Monitor  
**Date:** January 2026  
**Authors:** MTS3-MCTE Team Project - Energy G1  

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Proposed Solution](#2-proposed-solution)
3. [Target Audience Analysis](#3-target-audience-analysis)
4. [Comparison with Existing Solutions](#4-comparison-with-existing-solutions)
5. [User Needs & Pain Points](#5-user-needs--pain-points)
6. [User Flows](#6-user-flows)
7. [MVP Features](#7-mvp-features)
8. [Non-MVP / Future Features](#8-non-mvp--future-features)
9. [User Testing Methodology](#9-user-testing-methodology)
10. [Test Results](#10-test-results)
11. [Iterations & Improvements](#11-iterations--improvements)
12. [Conclusions](#12-conclusions)

---

## 1. Problem Definition

### 1.1 Context

The global push toward energy efficiency and sustainability has created an urgent need for granular energy consumption data at the household and building level. While smart meters are being deployed across Europe at an accelerating rate, the data they provide is limited to aggregate consumption figures.

### 1.2 Core Problem Statement

**Homeowners and building managers lack visibility into individual appliance energy consumption, preventing them from making informed decisions about energy optimization.**

### 1.3 Problem Dimensions

| Dimension | Description |
|-----------|-------------|
| **Visibility Gap** | Smart meters show total consumption but not per-appliance breakdown |
| **Cost Barrier** | Sub-metering solutions require €50-200 per appliance plus installation |
| **Technical Complexity** | Traditional disaggregation requires professional electrical work |
| **Actionability** | Users cannot identify which appliances to optimize |
| **Behavioral Change** | Without data, users cannot measure the impact of conservation efforts |

### 1.4 Market Need

- Rising electricity prices across Europe (40%+ increase 2021-2025)
- Regulatory push for energy efficiency (EU Energy Efficiency Directive)
- Consumer demand for sustainability insights
- Smart meter penetration exceeding 70% in many European countries

---

## 2. Proposed Solution

### 2.1 Solution Overview

NILM Energy Monitor is a web-based application that uses Non-Intrusive Load Monitoring (NILM) technology to disaggregate total household power consumption into individual appliance usage using deep learning.

### 2.2 How It Works

```
1. Data Ingestion
   Smart Meter → InfluxDB Time-Series Database

2. AI Processing
   Aggregate Power Window → PyTorch Seq2Point Model → Appliance Predictions

3. Visualization
   React Dashboard ← API ← Prediction Results
```

### 2.3 Technical Approach

- **Algorithm**: Seq2Point neural network architecture
- **Input**: Sliding window of aggregate power readings
- **Output**: Individual appliance power at window midpoint
- **Models**: CNN, Transformer, and UNet architectures trained per appliance

### 2.4 Supported Appliances (12 Categories)

| Category | Appliances |
|----------|-----------|
| **Kitchen** | Dishwasher, Oven, Stove, Range Hood |
| **Laundry** | Washing Machine, Dryer |
| **HVAC** | Heat Pump, Heat Pump Controller |
| **EV Charging** | Charging Station, Smappee Charger |
| **Utility** | Rainwater Pump, Garage Cabinet |

### 2.5 Key Differentiators

1. **Non-Intrusive**: No additional hardware required
2. **Real-Time**: Live disaggregation with <2 second latency
3. **Transparent AI**: Confidence indicators for all predictions
4. **Multi-Resolution**: Supports 1-second to 15-minute data granularity
5. **Multi-Tenant**: Supports multiple buildings per user

---

## 3. Target Audience Analysis

### 3.1 Persona 1: Energy-Conscious Homeowner

**Name:** Maria, 42  
**Occupation:** Marketing Manager  
**Location:** Suburban Belgium  

**Background:**
- Owns a 4-bedroom house with smart meter
- Concerned about rising energy costs
- Interested in sustainability
- Limited technical knowledge

**Goals:**
- Reduce monthly electricity bill
- Identify energy-hungry appliances
- Monitor heat pump efficiency
- Track impact of behavior changes

**Frustrations:**
- Cannot understand why bills are high
- No visibility into appliance consumption
- Utility dashboard is confusing
- Cannot justify solar panel investment

**Technology Comfort:** Medium  
**Device Usage:** Laptop (primary), Smartphone (secondary)

---

### 3.2 Persona 2: Research / Facility Manager

**Name:** Thomas, 38  
**Occupation:** Technical Manager  
**Location:** Ghent, Belgium  

**Background:**
- Manages technical data for buildings
- Reports to sustainability committee
- Has technical background
- Uses multiple monitoring systems

**Goals:**
- Optimize energy across buildings
- Generate monthly reports
- Identify inefficient equipment
- Meet ESG targets

**Frustrations:**
- Fragmented monitoring tools
- Manual report generation
- Cannot compare buildings
- Lack of predictive insights

**Technology Comfort:** High  
**Device Usage:** Desktop (primary), Tablet (on-site)

---

## 4. Comparison with Existing Solutions

### 4.1 Competitive Landscape

| Solution | Type | Hardware | Cost | Appliance-Level | Real-Time |
|----------|------|----------|------|-----------------|-----------|
| **NILM Monitor** | Software | None | €0-15/mo | Yes (AI) | Yes |
| **Sense Home** | Hardware | Required | €299 + app | Yes | Yes |
| **Emporia Vue** | Hardware | Required | €79-149 | Yes | Yes |
| **Utility Dashboard** | Software | None | Free | No | No |
| **Smappee** | Hardware | Required | €199-499 | Yes | Yes |

### 4.2 Competitive Advantages

1. **No Hardware Investment**: Works with existing smart meters
2. **AI Transparency**: Shows confidence levels for predictions
3. **Flexible Deployment**: Cloud or on-premise options
4. **Open Architecture**: API for integrations
5. **Multi-Building**: Single dashboard for multiple properties

### 4.3 Limitations vs. Hardware Solutions

- Prediction accuracy depends on training data quality
- Cannot detect very low-power devices (<50W)
- Requires sufficient historical data for calibration
- Overlapping appliance signatures may reduce accuracy

---

## 5. User Needs & Pain Points

### 5.1 Identified User Needs

| Priority | Need | Description |
|----------|------|-------------|
| **P0** | Appliance Breakdown | See which appliances consume the most energy |
| **P0** | Real-Time View | Current energy usage at a glance |
| **P1** | Historical Trends | Compare consumption over time |
| **P1** | Export Data | Download data for external analysis |
| **P2** | Alerts | Notifications for unusual consumption |
| **P2** | Multi-Building | Manage multiple properties |
| **P3** | Smart Home Integration | Connect with automation systems |

### 5.2 Pain Points Addressed

| Pain Point | How NILM Monitor Addresses It |
|------------|-------------------------------|
| No appliance visibility | AI disaggregation from aggregate data |
| High monitoring costs | Software-only solution |
| Complex installation | No hardware, web-based access |
| Unclear energy bills | Detailed breakdown by appliance |
| No actionable insights | Peak identification & recommendations |

---

## 6. User Flows

### 6.1 User Flow: First-Time User Onboarding

```
1. User lands on homepage
   ↓
2. Clicks "Get Started"
   ↓
3. Registration form (email, password)
   ↓
4. Email verification sent
   ↓
5. User verifies email, redirected to app
   ↓
6. Welcome modal with options:
   - "Try Demo Mode" → Load sample data
   - "Connect Building" → Building setup wizard
   ↓
7. If Demo: Dashboard loads with sample data
   If Connect: Building configuration form
   ↓
8. User sees Dashboard with data
```

### 6.2 User Flow: View Energy Dashboard

```
1. User logs into application
   ↓
2. Dashboard loads (default: last 7 days)
   ↓
3. User sees:
   - Total consumption chart
   - Top appliances breakdown
   - Current appliance status
   - Key metrics (peak, total, confidence)
   ↓
4. User can:
   - Change date range
   - Select different building
   - Click appliance for details
   - Refresh data
```

### 6.3 User Flow: Export Energy Data

```
1. User on Dashboard page
   ↓
2. Clicks "Export" dropdown button
   ↓
3. Selects export type:
   - "Export CSV" → Raw data
   - "Export Summary" → Text report
   ↓
4. Browser downloads file
   ↓
5. Success toast notification
```

### 6.4 User Flow: View Appliance Details

```
1. User on Dashboard or Appliances page
   ↓
2. Clicks on specific appliance card
   ↓
3. Modal opens with:
   - Appliance consumption chart
   - ON/OFF history
   - Average power usage
   - Confidence metrics
   ↓
4. User can close modal or navigate to full details
```

### 6.5 User Flow: Manage Buildings

```
1. User navigates to Settings → Buildings
   ↓
2. Views list of connected buildings
   ↓
3. Options:
   - Add new building
   - Edit building details
   - Remove building
   ↓
4. If Add: Form with name, address, timezone
   ↓
5. Building appears in building selector
```

---

## 7. MVP Features

### 7.1 Core Features (Implemented)

| Feature | Description | Status |
|---------|-------------|--------|
| **User Authentication** | Email/password login via Supabase | ✅ Complete |
| **Dashboard** | Real-time energy visualization | ✅ Complete |
| **Disaggregation** | AI-powered appliance breakdown | ✅ Complete |
| **Date Range Selection** | Filter data by time period | ✅ Complete |
| **Appliance Status** | ON/OFF detection with confidence | ✅ Complete |
| **Data Export (CSV)** | Download raw energy data | ✅ Complete |
| **Summary Reports** | Text-based consumption summary | ✅ Complete |
| **Multi-Building** | Support for multiple properties | ✅ Complete |
| **Demo Mode** | Sample data for evaluation | ✅ Complete |
| **Responsive Design** | Mobile and desktop support | ✅ Complete |

### 7.2 MVP Technical Requirements

| Requirement | Specification | Status |
|-------------|---------------|--------|
| Data Resolution | 15-minute intervals minimum | ✅ Met |
| API Latency | < 500ms for readings endpoint | ✅ Met |
| Model Inference | < 2 seconds per prediction | ✅ Met |
| Uptime | 99% availability | ✅ Met |
| Browser Support | Chrome, Firefox, Safari, Edge | ✅ Met |

---

## 8. Non-MVP / Future Features

### 8.1 Planned Features

| Feature | Priority | Description | Timeline |
|---------|----------|-------------|----------|
| **Push Notifications** | High | Alerts for anomalies | Q2 2026 |
| **Mobile App** | High | Native iOS/Android | Q2 2026 |
| **Smart Home Integration** | Medium | Home Assistant, IFTTT | Q3 2026 |
| **Predictive Analytics** | Medium | Forecast consumption | Q3 2026 |
| **Cost Calculation** | Medium | Apply electricity rates | Q2 2026 |
| **Comparison Reports** | Low | Compare periods/buildings | Q4 2026 |
| **API Access** | Low | Third-party integrations | Q4 2026 |

### 8.2 Technical Roadmap

1. **1-Second Resolution Support**: Enable high-frequency monitoring
2. **Edge Deployment**: On-premise inference for privacy
3. **Model Auto-Retraining**: Continuous learning from user data
4. **Multi-Language Support**: Internationalization
5. **Advanced Visualizations**: Power signature waveforms

---

## 9. User Testing Methodology

### 9.1 Testing Approach

**Type:** Moderated usability testing  
**Method:** Task-based evaluation with think-aloud protocol  
**Duration:** 30-45 minutes per session  
**Location:** Remote (video conferencing)

### 9.2 Participant Selection

| Criteria | Requirement |
|----------|-------------|
| Total Participants | 12 |
| Homeowners | 8 participants |
| Facility/Technical Managers | 4 participants |
| Age Range | 25-55 years |
| Technical Background | Mixed (low to high) |

### 9.3 Test Tasks

| Task ID | Description | Success Criteria |
|---------|-------------|------------------|
| T1 | Create account and log in | Complete registration flow |
| T2 | View energy dashboard | Navigate to dashboard |
| T3 | Change date range to last 30 days | Successfully filter data |
| T4 | Identify highest-consuming appliance | Correctly identify from chart |
| T5 | Export data as CSV | Download CSV file |
| T6 | Check if heat pump is currently ON | Find status in UI |
| T7 | View detailed appliance history | Open appliance modal |
| T8 | Generate summary report | Download summary file |

### 9.4 Metrics Collected

- Task completion rate
- Time on task
- Error rate
- System Usability Scale (SUS) score
- Net Promoter Score (NPS)
- Qualitative feedback

---

## 10. Test Results

### 10.1 Quantitative Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **SUS Score** | 78/100 | ≥70 | ✅ Pass |
| **Task Completion** | 94% | ≥90% | ✅ Pass |
| **Avg. Task Time** | 45s | <60s | ✅ Pass |
| **Error Rate** | 6% | <10% | ✅ Pass |
| **User Satisfaction** | 4.2/5 | ≥4.0 | ✅ Pass |
| **NPS** | 42 | ≥30 | ✅ Pass |

### 10.2 Task-Level Results

| Task | Completion | Avg. Time | Issues |
|------|------------|-----------|--------|
| T1: Registration | 100% | 62s | Email verification unclear |
| T2: Dashboard | 100% | 8s | None |
| T3: Date Range | 92% | 35s | Picker confusion |
| T4: Top Appliance | 100% | 12s | None |
| T5: Export CSV | 83% | 55s | Button not visible |
| T6: Appliance Status | 92% | 20s | None |
| T7: Appliance Details | 100% | 15s | None |
| T8: Summary Report | 83% | 48s | Confused with CSV |

### 10.3 Qualitative Findings

**Positive Feedback:**
- "The dashboard is very clean and easy to understand"
- "I love seeing which appliances are currently on"
- "The confidence indicators help me trust the data"
- "Chart colors are easy to distinguish"

**Areas for Improvement:**
- "I didn't notice the export button at first"
- "The date picker is a bit confusing"
- "What does 'confidence' mean exactly?"
- "Would be nice to see cost estimates"

### 10.4 Usability Issues Identified

| Issue | Severity | Frequency | Resolution |
|-------|----------|-----------|------------|
| Export button not visible | High | 4/12 | Moved to prominent position |
| Date picker UX confusion | Medium | 3/12 | Redesigned picker |
| Confidence meaning unclear | Medium | 5/12 | Added tooltip explanation |
| No onboarding guidance | Medium | 6/12 | Added welcome modal |
| Summary vs CSV confusion | Low | 2/12 | Clearer labels |

---

## 11. Iterations & Improvements

### 11.1 Post-Testing Changes

| Issue | Change Implemented | Result |
|-------|-------------------|--------|
| Export visibility | Added prominent "Export" dropdown in header | 100% discovery in re-test |
| Date picker | Replaced with preset buttons + custom range | Reduced errors to 0% |
| Confidence unclear | Added info tooltip with explanation | Users reported clarity |
| No onboarding | Implemented welcome modal with demo option | First-time experience improved |
| Loading states | Added skeleton loaders | Perceived performance improved |

### 11.2 Technical Improvements

1. **API Optimization**: Reduced average response time from 450ms to 280ms
2. **Caching Layer**: Added Redis caching for frequently accessed data
3. **Error Handling**: Improved error messages with actionable suggestions
4. **Mobile Responsiveness**: Fixed layout issues on tablet devices

### 11.3 Design Refinements

- Enhanced color contrast for accessibility (WCAG AA compliance)
- Increased button sizes for touch targets
- Improved chart legend readability
- Added empty state designs

---

## 12. Conclusions

### 12.1 Project Outcomes

The NILM Energy Monitor successfully delivers on its core promise of providing appliance-level energy insights without additional hardware. User testing validated the application's usability and value proposition.

### 12.2 Key Achievements

- Functional end-to-end NILM pipeline
- SUS score of 78 (Good usability)
- 94% task completion rate
- Production-ready deployment
- Positive user feedback on core functionality

### 12.3 Lessons Learned

1. **AI Transparency Matters**: Users appreciated confidence indicators
2. **Onboarding is Critical**: First-time experience significantly impacts adoption
3. **Simplicity Wins**: Clean, focused UI outperformed feature-rich alternatives
4. **Iterative Testing Works**: Post-testing improvements measurably increased usability

### 12.4 Recommendations for Future Development

1. Prioritize cost calculation feature (high user demand)
2. Develop push notification system for engagement
3. Consider native mobile app for convenience
4. Expand appliance model coverage
5. Implement user-driven model calibration

---

*Document Version: Final | Last Updated: January 2026*
