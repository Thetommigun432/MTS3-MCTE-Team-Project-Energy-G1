# NILM Energy Monitor - User Manual

**Version:** 1.0 Final  
**Product:** NILM Energy Monitor Web Application  
**Date:** January 2026  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Getting Started](#3-getting-started)
4. [Interface Overview](#4-interface-overview)
5. [Main Features](#5-main-features)
6. [Common Use Cases](#6-common-use-cases)
7. [Advanced Features](#7-advanced-features)
8. [Troubleshooting](#8-troubleshooting)
9. [Frequently Asked Questions](#9-frequently-asked-questions)
10. [Support & Contact](#10-support--contact)

---

## 1. Introduction

### 1.1 Welcome to NILM Energy Monitor

NILM Energy Monitor is a web-based application that helps you understand your household or building energy consumption at the appliance level. Using advanced AI technology, it breaks down your total electricity usage to show you exactly which appliances are consuming power.

### 1.2 What is NILM?

NILM stands for **Non-Intrusive Load Monitoring**. This technology allows us to identify individual appliance consumption from your smart meter's total power reading—without installing any additional sensors on your appliances.

### 1.3 Key Benefits

- **See the Invisible**: Understand which appliances use the most electricity
- **Save Money**: Identify opportunities to reduce energy consumption
- **No Hardware Required**: Works with your existing smart meter data
- **Real-Time Insights**: View current consumption instantly
- **Historical Analysis**: Track trends over days, weeks, or months

### 1.4 What This Manual Covers

This manual will guide you through:
- Setting up your account and first building
- Navigating the application interface
- Using all major features
- Solving common issues

---

## 2. System Requirements

### 2.1 Supported Browsers

NILM Energy Monitor is a web application that runs in your browser. For the best experience, use one of the following:

| Browser | Minimum Version | Recommended |
|---------|-----------------|-------------|
| Google Chrome | 90+ | Latest |
| Mozilla Firefox | 88+ | Latest |
| Microsoft Edge | 90+ | Latest |
| Safari | 14+ | Latest |

### 2.2 Device Requirements

| Device Type | Screen Size | Notes |
|-------------|-------------|-------|
| Desktop/Laptop | 1024px+ | Full experience |
| Tablet | 768px+ | Full experience |
| Smartphone | 320px+ | Responsive layout |

### 2.3 Internet Connection

A stable internet connection is required. Minimum recommended speed: 5 Mbps download.

### 2.4 Data Requirements

To use NILM Energy Monitor with real data, you need:
- An active smart meter in your building
- Data available through your utility provider or a compatible data source
- Building configured in the system (done by administrator or during onboarding)

**Note:** You can explore the application using Demo Mode without any data connection.

---

## 3. Getting Started

### 3.1 Creating an Account

1. Navigate to the NILM Energy Monitor website
2. Click **"Get Started"** or **"Sign Up"** on the homepage
3. Enter your email address and create a password
4. Click **"Create Account"**
5. Check your email inbox for a verification message
6. Click the verification link in the email
7. You will be redirected to the application

**Password Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one number

### 3.2 First Login

After verifying your email:

1. Enter your email and password on the login page
2. Click **"Sign In"**
3. A Welcome Modal will appear with two options:
   - **"Try Demo Mode"**: Explore with sample data (recommended for first-time users)
   - **"Connect Building"**: Set up your actual building

### 3.3 Demo Mode

Demo Mode allows you to explore all features using pre-loaded sample data:

1. Click **"Try Demo Mode"** in the welcome modal
2. The dashboard will load with example energy data
3. All features work normally with this sample data
4. Switch to real data anytime via Settings

**Note:** In Demo Mode, you will see a badge indicating "Demo Data" in the top bar.

### 3.4 Setting Up Your Building

To connect your actual building:

1. Click **"Connect Building"** or go to **Settings → Buildings**
2. Click **"Add Building"**
3. Fill in the building details:
   - **Name**: A friendly name (e.g., "Home", "Office")
   - **Address**: Street address (optional)
   - **Timezone**: Select your local timezone
4. Click **"Save Building"**
5. Follow the data connection instructions provided

---

## 4. Interface Overview

### 4.1 Main Navigation

The application is organized into the following sections:

| Menu Item | Description |
|-----------|-------------|
| **Dashboard** | Main energy overview and charts |
| **Appliances** | List of all detected appliances |
| **Reports** | Generate and download reports |
| **Buildings** | Manage your properties |
| **Settings** | Account and application preferences |
| **Help** | Documentation and support |

### 4.2 Dashboard Layout

The Dashboard is divided into several areas:

```
┌─────────────────────────────────────────────────────┐
│  TOP BAR: Building selector, Date range, Export    │
├─────────────────────────────────────────────────────┤
│  METRIC CARDS: Peak Load | Total Energy | Top      │
│                Appliance | Model Confidence        │
├─────────────────────────────────────────────────────┤
│  MAIN CHART: Total consumption over time           │
├───────────────────────┬─────────────────────────────┤
│  APPLIANCE BREAKDOWN  │  WHAT'S ON NOW             │
│  (stacked area chart) │  (current status)          │
└───────────────────────┴─────────────────────────────┘
```

### 4.3 Key Interface Elements

#### Building Selector
Located in the top bar. Click to switch between multiple buildings if you have more than one configured.

#### Date Range Picker
Choose the time period for data display:
- **Today**: Current day only
- **7 Days**: Last week
- **30 Days**: Last month
- **Custom**: Select specific start and end dates

#### Refresh Button
Click the refresh icon (🔄) to fetch the latest data from the server.

#### Export Menu
Click **"Export"** to download data:
- **Export CSV**: Raw data in spreadsheet format
- **Export Summary**: Text summary report

### 4.4 Understanding Cards

Metric cards display key information:

| Card | What It Shows |
|------|---------------|
| **Peak Load** | Highest power consumption in the period |
| **Total Energy** | Sum of all energy consumed (kWh) |
| **Top Consumer** | Appliance using the most energy |
| **Model Confidence** | AI prediction reliability score |

---

## 5. Main Features

### 5.1 Real-Time Consumption Chart

The main chart displays total power consumption over your selected time period.

**Reading the Chart:**
- **X-axis**: Time (date and hour)
- **Y-axis**: Power consumption in kilowatts (kW)
- **Hover**: Shows exact value at any point

**Interacting with the Chart:**
- Hover over any point to see detailed values
- The chart automatically scales to your data

### 5.2 Appliance Breakdown Chart

The stacked area chart shows how total consumption is divided among appliances.

**Understanding the Colors:**
Each color represents a different appliance. The legend below the chart shows which color corresponds to which appliance.

**Reading the Chart:**
- The total height at any point equals total consumption
- Each colored band shows one appliance's contribution
- Larger bands indicate higher-consuming appliances

### 5.3 What's ON Now Panel

This panel shows the current status of your appliances:

| Column | Meaning |
|--------|---------|
| **Appliance** | Appliance name and icon |
| **Status** | ON (green) or OFF (gray) |
| **Power** | Current estimated power in kW |
| **Confidence** | How sure the AI is about this reading |

**Status Indicators:**
- 🟢 **ON**: Appliance is currently consuming power
- ⚫ **OFF**: Appliance is not consuming power

### 5.4 Appliance Details

Click on any appliance to view detailed information:

**Detail Modal Contents:**
- Consumption chart for this appliance only
- ON/OFF history over the selected period
- Average power when running
- Total energy consumed
- Peak usage time
- Confidence metrics

### 5.5 Changing Date Range

To view different time periods:

1. Click the date range button in the top bar
2. Select a preset (Today, 7 Days, 30 Days) OR
3. Click "Custom" to choose specific dates
4. Data will automatically refresh

### 5.6 Exporting Data

#### Export to CSV
1. Click **"Export"** in the top bar
2. Select **"Export CSV"**
3. File downloads with columns: Timestamp, Aggregate, [each appliance]

#### Export Summary
1. Click **"Export"** in the top bar
2. Select **"Export Summary"**
3. Text file downloads with key metrics and insights

---

## 6. Common Use Cases

### 6.1 Finding Your Biggest Energy Consumer

**Goal:** Identify which appliance uses the most electricity

**Steps:**
1. Set the date range to **30 Days** for a representative period
2. Look at the **Top Consumer** metric card—this shows the answer immediately
3. Check the **Appliance Breakdown** chart to see relative consumption
4. Click on the top appliance for detailed analysis

### 6.2 Checking If an Appliance is Running

**Goal:** Verify if a specific appliance (e.g., heat pump) is currently ON

**Steps:**
1. Look at the **What's ON Now** panel on the Dashboard
2. Find the appliance in the list
3. Check its status indicator (green = ON, gray = OFF)
4. The Power column shows current consumption

### 6.3 Tracking Daily Energy Usage

**Goal:** See how much energy you used today

**Steps:**
1. Click the date range and select **"Today"**
2. Look at the **Total Energy** metric card (in kWh)
3. View the main chart to see consumption over the day
4. Identify peak usage times for potential savings

### 6.4 Comparing Weeks or Months

**Goal:** Compare energy usage between different periods

**Steps:**
1. Set the date range to your first period (e.g., last 7 days)
2. Note the Total Energy value
3. Export a summary report for this period
4. Change the date range to the comparison period
5. Compare the values

### 6.5 Generating a Monthly Report

**Goal:** Create a report for your records or to share

**Steps:**
1. Set the date range to the full month
2. Click **"Export"** → **"Export Summary"**
3. Save the text file
4. Optionally, also export the CSV for detailed data

---

## 7. Advanced Features

### 7.1 Managing Multiple Buildings

If you manage multiple properties:

**Adding a Building:**
1. Go to **Settings** → **Buildings**
2. Click **"Add Building"**
3. Enter building details
4. Save

**Switching Buildings:**
1. Click the building name in the top bar
2. Select a different building from the dropdown
3. Dashboard updates to show that building's data

### 7.2 Understanding Model Confidence

The AI models provide a confidence score with each prediction:

| Confidence Level | Meaning |
|-----------------|---------|
| **High (80-100%)** | Predictions are very reliable |
| **Medium (50-79%)** | Predictions are generally accurate |
| **Low (<50%)** | Predictions may be less accurate |

**Factors Affecting Confidence:**
- Similarity to training data
- Overlapping appliance signatures
- Unusual usage patterns
- Data quality

### 7.3 Appliance Categories

Appliances are organized into categories:

| Category | Examples |
|----------|----------|
| **Kitchen** | Dishwasher, Oven, Stove, Range Hood |
| **Laundry** | Washing Machine, Dryer |
| **HVAC** | Heat Pump, Heat Pump Controller |
| **EV Charging** | Charging Station, Smappee Charger |
| **Utility** | Rainwater Pump |

### 7.4 Theme Settings

Switch between dark and light mode:

1. Click your profile icon in the top bar
2. Select **"Toggle Theme"** or go to **Settings**
3. Choose your preferred theme

---

## 8. Troubleshooting

### 8.1 Common Issues and Solutions

#### "No Data Available"

**Possible Causes:**
- No data for selected date range
- Building not connected
- Data sync in progress

**Solutions:**
1. Try selecting a different date range
2. Check your building configuration in Settings
3. Wait a few minutes and refresh
4. Contact support if the issue persists

#### Dashboard Not Loading

**Solutions:**
1. Refresh the page (F5 or Ctrl+R)
2. Clear browser cache
3. Try a different browser
4. Check your internet connection

#### Export Button Not Working

**Solutions:**
1. Ensure pop-up blocker is disabled for this site
2. Check if data is available (cannot export empty data)
3. Try a different browser

#### Login Issues

**"Invalid Credentials":**
- Double-check email and password
- Use "Forgot Password" to reset

**"Email Not Verified":**
- Check your email for verification link
- Check spam/junk folder
- Request new verification email

#### Charts Not Displaying

**Solutions:**
1. Refresh the page
2. Ensure JavaScript is enabled
3. Update your browser to the latest version
4. Disable browser extensions that might interfere

### 8.2 Error Messages

| Error | Meaning | Action |
|-------|---------|--------|
| "Session Expired" | Login timed out | Log in again |
| "Network Error" | Connection issue | Check internet, retry |
| "Data Unavailable" | No data for query | Change date range |
| "Rate Limited" | Too many requests | Wait a moment, retry |

### 8.3 Performance Tips

- Use shorter date ranges for faster loading
- Close other browser tabs if experiencing slowness
- Modern browsers (Chrome, Firefox) perform best
- Mobile data may be slower than WiFi

---

## 9. Frequently Asked Questions

### General Questions

**Q: How accurate are the appliance predictions?**
A: Accuracy varies by appliance and usage patterns. High-consumption appliances (heat pumps, ovens) are typically most accurate. Confidence scores indicate prediction reliability.

**Q: Does NILM work with all appliances?**
A: NILM works best with appliances that have distinctive power signatures. Very low-power devices (<50W) may not be detected.

**Q: How far back can I view historical data?**
A: This depends on your data source. Typically, up to 12 months of historical data is available.

### Technical Questions

**Q: How does NILM technology work?**
A: NILM uses machine learning models trained on appliance power signatures. When you use your dishwasher, it creates a unique pattern in total consumption. Our AI recognizes these patterns.

**Q: Is my data secure?**
A: Yes. All data is encrypted in transit (HTTPS) and at rest. We follow industry best practices for data security.

**Q: Can I use NILM with solar panels?**
A: Yes. The system handles homes with solar generation, though complex scenarios may affect accuracy.

**Q: Why is my confidence score low?**
A: Low confidence can occur when multiple appliances run simultaneously, when an appliance is used in an unusual way, or when the usage pattern differs from training data.

### Account Questions

**Q: How do I delete my account?**
A: Go to Settings → Account → Delete Account. This action is irreversible.

**Q: Can I share access with family members?**
A: Multi-user access is on our roadmap. Currently, share your login credentials carefully if needed.

**Q: How do I change my email address?**
A: Go to Settings → Account → Edit Profile. Verify the new email address.

---

## 10. Support & Contact

### 10.1 Getting Help

If you need assistance:

1. **Check this Manual**: Most answers are here
2. **In-App Help**: Click the Help icon (?) in the navigation
3. **FAQ Section**: Common questions answered above
4. **Contact Support**: See below

### 10.2 Contact Information

**Email Support:**
support@nilm-energy-monitor.example.com

**Support:**
Please contact the project administration or your system administrator for support.
If you encounter a bug:

1. Note what you were doing when it occurred
2. Take a screenshot if possible
3. Note your browser and device
4. Send details to support email

### 10.4 Feature Requests

We welcome suggestions! Send feature requests to:
feedback@nilm-energy-monitor.example.com

### 10.5 Version Information
Please submit them to the project repository maintainers.
**Release Date:** January 2026  
**Last Updated:** January 24, 2026

---

*Thank you for using NILM Energy Monitor!*

---

*© 2026 MTS3-MCTE Team Project - Energy G1. All rights reserved.*
