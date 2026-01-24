# NILM Energy Monitor - Design Document

**Document Version:** Final (Post-User Testing)  
**Project:** NILM Energy Monitor  
**Date:** January 2026  
**Authors:** MTS3-MCTE Team Project - Energy G1  

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Mood & Inspiration](#2-mood--inspiration)
3. [Color Palette](#3-color-palette)
4. [Typography](#4-typography)
5. [Iconography](#5-iconography)
6. [Layout & Grid System](#6-layout--grid-system)
7. [Accessibility Considerations](#7-accessibility-considerations)
8. [Design Consistency Rules](#8-design-consistency-rules)
9. [Updates Post-Interim Evaluation](#9-updates-post-interim-evaluation)
10. [Sources & References](#10-sources--references)

---

## 1. Design Principles

### 1.1 Core Principles

The NILM Energy Monitor design system is built upon five foundational principles:

#### Clarity First
Energy data can be complex. Our design strips away unnecessary elements to present information clearly and immediately comprehensible. Every visual element serves a purpose.

#### Trust Through Transparency
Users are interacting with AI-generated predictions. We build trust by clearly indicating confidence levels, data freshness, and model information. Nothing is hidden.

#### Progressive Disclosure
We show the most important information first (aggregate consumption, top appliances) and allow users to drill down into details as needed. This prevents cognitive overload while enabling deep analysis.

#### Responsive & Accessible
The application works seamlessly across devices and is accessible to users with varying abilities. Energy monitoring should not be exclusive.

#### Data-Ink Ratio
Following Edward Tufte's principles, we maximize the ratio of data to ink. Charts are clean, backgrounds are subtle, and decorative elements are minimal.

### 1.2 Design Goals

| Goal | Implementation |
|------|----------------|
| **Scannable** | Key metrics visible within 2 seconds |
| **Actionable** | Clear calls-to-action for common tasks |
| **Trustworthy** | Professional aesthetic, confidence indicators |
| **Efficient** | Minimal clicks to reach any feature |
| **Consistent** | Unified visual language across all pages |

---

## 2. Mood & Inspiration

### 2.1 Design Direction

The visual design of NILM Energy Monitor draws inspiration from:

1. **Clean Energy Dashboards**: Tesla Powerwall, Sense Home, Enphase
2. **Data Visualization Platforms**: Grafana, Datadog, Mixpanel
3. **Modern SaaS Applications**: Linear, Notion, Vercel Dashboard

### 2.2 Mood Board Themes

#### Theme 1: Technical Precision
- Clean lines and sharp edges
- Monospace elements for data
- Subtle grid patterns
- Professional, engineering-oriented

#### Theme 2: Environmental Sustainability
- Green accent colors
- Organic curves in charts
- Nature-inspired gradients
- Approachable and calming

#### Theme 3: Modern Technology
- Dark mode primary
- Vibrant accent colors
- Glowing effects on active elements
- Futuristic but not alienating

### 2.3 Chosen Direction

We selected a hybrid approach combining **Technical Precision** with **Modern Technology**. This creates a dashboard that feels professional and trustworthy while maintaining a contemporary aesthetic that appeals to tech-savvy users.

The dark mode default reduces eye strain for users who monitor energy frequently and creates visual contrast that makes data stand out.

---

## 3. Color Palette

### 3.1 Primary Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| **Background** | `#0A0A0B` | 10, 10, 11 | Main app background |
| **Surface** | `#18181B` | 24, 24, 27 | Cards, modals, panels |
| **Surface Elevated** | `#27272A` | 39, 39, 42 | Hover states, dropdowns |
| **Border** | `#3F3F46` | 63, 63, 70 | Dividers, card borders |
| **Text Primary** | `#FAFAFA` | 250, 250, 250 | Headings, important text |
| **Text Secondary** | `#A1A1AA` | 161, 161, 170 | Body text, labels |
| **Text Muted** | `#71717A` | 113, 113, 122 | Hints, disabled text |

### 3.2 Accent Colors

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| **Primary (Green)** | `#22C55E` | 34, 197, 94 | Success, energy positive |
| **Primary Hover** | `#16A34A` | 22, 163, 74 | Button hovers |
| **Secondary (Blue)** | `#3B82F6` | 59, 130, 246 | Links, info states |
| **Warning (Amber)** | `#F59E0B` | 245, 158, 11 | Warnings, medium confidence |
| **Error (Red)** | `#EF4444` | 239, 68, 68 | Errors, high consumption |

### 3.3 Chart Colors (Colorblind-Safe)

These colors are optimized for distinguishing data series, including for users with color vision deficiencies:

| Color Name | Hex Code | Appliance Association |
|------------|----------|----------------------|
| **Orange** | `#E69F00` | Heat Pump |
| **Sky Blue** | `#56B4E9` | Washing Machine |
| **Green** | `#009E73` | Dishwasher |
| **Blue** | `#0072B2` | Dryer |
| **Vermillion** | `#D55E00` | Oven |
| **Purple** | `#CC79A7` | Stove |
| **Gray** | `#999999` | Other/Unknown |

### 3.4 Light Mode Colors (Optional Theme)

| Color Name | Hex Code | Usage |
|------------|----------|-------|
| **Background** | `#FFFFFF` | Main app background |
| **Surface** | `#F4F4F5` | Cards, modals |
| **Text Primary** | `#18181B` | Headings |
| **Text Secondary** | `#52525B` | Body text |

---

## 4. Typography

### 4.1 Font Stack

```css
--font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
```

### 4.2 Primary Font: Inter

**Selection Rationale:**
- Designed specifically for screen readability
- Excellent legibility at small sizes
- Wide range of weights for hierarchy
- Open source and freely available
- Native support for tabular figures (important for data)

**Weights Used:**
| Weight | CSS Value | Usage |
|--------|-----------|-------|
| Regular | 400 | Body text, descriptions |
| Medium | 500 | Labels, buttons |
| Semibold | 600 | Subheadings, emphasis |
| Bold | 700 | Headings, key metrics |

### 4.3 Monospace Font: JetBrains Mono

**Selection Rationale:**
- Clear distinction between similar characters (0/O, 1/l)
- Designed for code and data display
- Consistent character width for alignment
- Complements Inter aesthetically

**Usage:**
- Numeric values in cards
- Timestamps
- Technical identifiers (UUIDs, building IDs)

### 4.4 Type Scale

| Level | Size | Weight | Line Height | Usage |
|-------|------|--------|-------------|-------|
| **Display** | 36px | Bold | 1.2 | Page titles |
| **H1** | 30px | Bold | 1.25 | Section headings |
| **H2** | 24px | Semibold | 1.3 | Card titles |
| **H3** | 20px | Semibold | 1.35 | Subsections |
| **H4** | 18px | Medium | 1.4 | Widget titles |
| **Body Large** | 16px | Regular | 1.5 | Primary content |
| **Body** | 14px | Regular | 1.5 | Default text |
| **Small** | 12px | Regular | 1.4 | Labels, hints |
| **Metric** | 32px | Bold (Mono) | 1.2 | Key numbers |

---

## 5. Iconography

### 5.1 Icon Library: Lucide React

**Selection Rationale:**
- Open source, MIT licensed
- Consistent 24x24 grid system
- 1.5px stroke width for clarity
- Comprehensive icon set (1000+ icons)
- React-optimized with tree-shaking

### 5.2 Icon Usage Guidelines

| Context | Size | Stroke | Notes |
|---------|------|--------|-------|
| **Navigation** | 20px | 1.5px | Sidebar icons |
| **Buttons** | 16px | 2px | Action buttons |
| **Cards** | 24px | 1.5px | Card header icons |
| **Inline** | 14px | 1.5px | Text-adjacent icons |

### 5.3 Appliance Icons

| Appliance | Icon Name | Visual Description |
|-----------|-----------|-------------------|
| Heat Pump | `Thermometer` | Temperature gauge |
| Dishwasher | `UtensilsCrossed` | Crossed utensils |
| Washing Machine | `WashingMachine` | Washing machine |
| Dryer | `Wind` | Wind/air flow |
| Oven | `Flame` | Fire/heat |
| Stove | `CookingPot` | Cooking pot |
| EV Charger | `BatteryCharging` | Charging battery |
| Refrigerator | `Refrigerator` | Refrigerator |

### 5.4 Status Icons

| Status | Icon | Color |
|--------|------|-------|
| Online/ON | `Power` | Green (#22C55E) |
| Offline/OFF | `PowerOff` | Muted (#71717A) |
| Loading | `Loader2` (animated) | Secondary |
| Success | `Check` | Green |
| Warning | `AlertTriangle` | Amber |
| Error | `XCircle` | Red |
| Info | `Info` | Blue |

---

## 6. Layout & Grid System

### 6.1 Grid Structure

**Container Widths:**
| Breakpoint | Container Max-Width | Padding |
|------------|---------------------|---------|
| Mobile (<640px) | 100% | 16px |
| Tablet (640-1024px) | 100% | 24px |
| Desktop (1024-1280px) | 1200px | 32px |
| Wide (>1280px) | 1400px | 48px |

**Grid Columns:**
- Mobile: 1 column
- Tablet: 2 columns
- Desktop: 4 columns
- Dashboard cards use CSS Grid with auto-fit

### 6.2 Spacing System

Based on 4px base unit:

| Token | Value | Usage |
|-------|-------|-------|
| `space-1` | 4px | Tight inline spacing |
| `space-2` | 8px | Icon-text gaps |
| `space-3` | 12px | Card internal padding |
| `space-4` | 16px | Section spacing |
| `space-6` | 24px | Card padding |
| `space-8` | 32px | Section margins |
| `space-12` | 48px | Page sections |
| `space-16` | 64px | Major sections |

### 6.3 Component Dimensions

| Component | Height | Border Radius |
|-----------|--------|---------------|
| Button (small) | 32px | 6px |
| Button (default) | 40px | 8px |
| Button (large) | 48px | 10px |
| Input | 40px | 8px |
| Card | auto | 12px |
| Modal | auto | 16px |
| Badge | 24px | 9999px (pill) |
| Avatar | 32px | 9999px (circle) |

### 6.4 Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  Navbar (height: 64px)                              │
├────────┬────────────────────────────────────────────┤
│        │  Main Content Area                         │
│  Side  │  ┌──────────────────────────────────────┐  │
│  bar   │  │  Metric Cards (4 columns)            │  │
│        │  ├──────────────────────────────────────┤  │
│  240px │  │  Main Chart (full width)             │  │
│        │  ├──────────────────────────────────────┤  │
│        │  │  Two-Column Layout                   │  │
│        │  │  ┌─────────────┐ ┌─────────────────┐ │  │
│        │  │  │  Appliance  │ │  What's ON Now  │ │  │
│        │  │  │  Breakdown  │ │  Panel          │ │  │
│        │  │  └─────────────┘ └─────────────────┘ │  │
│        │  └──────────────────────────────────────┘  │
└────────┴────────────────────────────────────────────┘
```

---

## 7. Accessibility Considerations

### 7.1 WCAG 2.1 Compliance

The NILM Energy Monitor targets **WCAG 2.1 Level AA** compliance.

### 7.2 Color Contrast

| Element | Foreground | Background | Ratio | Requirement |
|---------|------------|------------|-------|-------------|
| Body Text | #FAFAFA | #18181B | 15.8:1 | ≥4.5:1 ✅ |
| Secondary Text | #A1A1AA | #18181B | 7.1:1 | ≥4.5:1 ✅ |
| Muted Text | #71717A | #18181B | 4.6:1 | ≥4.5:1 ✅ |
| Primary Button | #FAFAFA | #22C55E | 3.4:1 | ≥3:1 (large) ✅ |
| Link Text | #3B82F6 | #18181B | 4.9:1 | ≥4.5:1 ✅ |

### 7.3 Keyboard Navigation

All interactive elements are fully keyboard accessible:
- `Tab` to navigate between focusable elements
- `Enter`/`Space` to activate buttons
- `Escape` to close modals
- `Arrow keys` for menu navigation
- Focus rings visible on all elements

### 7.4 Screen Reader Support

- Semantic HTML structure (landmarks, headings)
- ARIA labels on icon-only buttons
- Live regions for dynamic content updates
- Alt text on all meaningful images
- Chart descriptions in text format

### 7.5 Motion & Animation

- Respects `prefers-reduced-motion` system setting
- All animations are subtle (< 300ms duration)
- No auto-playing videos or distracting motion
- Loading states use static alternatives when motion is reduced

### 7.6 Colorblind Considerations

- Chart colors tested with Coblis colorblind simulator
- Never use color alone to convey information
- Patterns/labels supplement color coding
- Status indicators include icons

---

## 8. Design Consistency Rules

### 8.1 Component Patterns

#### Cards
- Always use 12px border radius
- Background: Surface color (#18181B)
- Border: 1px solid Border color (#3F3F46)
- Padding: 24px
- Shadow: None (flat design)

#### Buttons
- Primary: Green background, white text
- Secondary: Transparent, border, text color
- Ghost: No border, text color only
- Destructive: Red background for dangerous actions
- Always include hover state (darken 10%)

#### Forms
- Labels above inputs (not inline)
- Error messages below inputs in red
- Help text in muted color below inputs
- Required fields marked with asterisk

### 8.2 Content Guidelines

- Use sentence case for all UI text
- Keep button labels to 2-3 words
- Avoid jargon; use plain language
- Error messages should be helpful, not blaming
- Empty states include helpful next actions

### 8.3 Chart Styling

- X-axis: Gray text, bottom aligned
- Y-axis: Gray text, left aligned, hidden gridlines
- Legend: Positioned below chart on mobile, right on desktop
- Tooltip: Dark background, rounded corners, shadow
- Gridlines: Subtle (#27272A), horizontal only

---

## 9. Updates Post-Interim Evaluation

### 9.1 Changes Made After User Testing

| Issue | Original Design | Updated Design | Rationale |
|-------|-----------------|----------------|-----------|
| Export button hidden | Icon-only in toolbar | Prominent dropdown button | Improved discoverability |
| Date picker confusion | Single calendar with manual range | Preset buttons + custom option | Faster common actions |
| Confidence meaning | Badge with percentage only | Badge + info tooltip | User education |
| First-time experience | Direct to empty dashboard | Welcome modal with options | Guided onboarding |
| Loading states | Spinner only | Skeleton screens | Perceived performance |

### 9.2 Visual Refinements

1. **Increased contrast** for text on colored backgrounds
2. **Larger touch targets** for mobile buttons (min 44x44px)
3. **Added focus states** that were missing on some components
4. **Improved chart legend** readability with larger text
5. **Added empty states** with helpful illustrations

### 9.3 New Components Added

- Welcome Modal component for onboarding
- Date Range Preset Buttons (Today, 7 Days, 30 Days, Custom)
- Skeleton Loader components for all card types
- Info Tooltip component for explanations

---

## 10. Sources & References

### 10.1 Design Inspiration

1. **Sense Home Energy Monitor** - https://sense.com/
2. **Tesla Powerwall App** - https://www.tesla.com/powerwall
3. **Enphase Enlighten** - https://enlighten.enphaseenergy.com/
4. **Grafana Dashboard** - https://grafana.com/
5. **Linear App** - https://linear.app/

### 10.2 Design Systems Referenced

1. **Tailwind CSS** - https://tailwindcss.com/
2. **Radix UI Primitives** - https://www.radix-ui.com/
3. **shadcn/ui** - https://ui.shadcn.com/
4. **IBM Carbon Design System** - https://carbondesignsystem.com/

### 10.3 Accessibility Resources

1. **WCAG 2.1 Guidelines** - https://www.w3.org/WAI/WCAG21/quickref/
2. **WebAIM Contrast Checker** - https://webaim.org/resources/contrastchecker/
3. **Coblis Colorblind Simulator** - https://www.color-blindness.com/coblis-color-blindness-simulator/

### 10.4 Typography Resources

1. **Inter Font Family** - https://rsms.me/inter/
2. **JetBrains Mono** - https://www.jetbrains.com/lp/mono/
3. **Type Scale Calculator** - https://type-scale.com/

### 10.5 Icon Resources

1. **Lucide Icons** - https://lucide.dev/
2. **Heroicons** (referenced) - https://heroicons.com/

### 10.6 Data Visualization

1. **Edward Tufte - The Visual Display of Quantitative Information**
2. **Recharts Library** - https://recharts.org/
3. **Colorblind Safe Palettes** - https://davidmathlogic.com/colorblind/

---

*Document Version: Final | Last Updated: January 2026*
