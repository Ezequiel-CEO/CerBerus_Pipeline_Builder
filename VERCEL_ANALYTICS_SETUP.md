# Vercel Web Analytics Setup Guide

This document explains how to set up and use Vercel Web Analytics with the CerBerus Pipeline Builder dashboard.

## Overview

Vercel Web Analytics has been integrated into the Gradio-based Pipeline Builder dashboard. The analytics script is automatically injected into the web interface when enabled, allowing you to track visitor statistics, page views, and user interactions.

## Prerequisites

- A Vercel account (free tier available at [vercel.com](https://vercel.com))
- Your application deployed to Vercel (or running locally with Vercel CLI)

## Setup Instructions

### 1. Enable Analytics in Vercel Dashboard

1. Log in to your [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project
3. Navigate to the **Analytics** tab
4. Click the **Enable** button
5. This will add the necessary routes (`/_vercel/insights/*`) to your deployment

### 2. Configure Environment Variables

Create a `.env` file in your project root (you can copy from `.env.example`):

```bash
cp .env.example .env
```

Edit the `.env` file and set:

```env
# Enable Vercel Analytics
VERCEL_ANALYTICS_ENABLED=true

# Optional: Custom Analytics ID (usually not needed)
# VERCEL_ANALYTICS_ID=your-analytics-id
```

### 3. Run the Dashboard

Start the dashboard as usual:

```bash
# Using CLI
python -m cerberus_api.pipeline_builder.cli dashboard --port 7860

# Or directly
python ui/pipeline_dashboard.py
```

The analytics script will be automatically injected when `VERCEL_ANALYTICS_ENABLED=true`.

### 4. Verify Analytics is Working

1. Open your dashboard in a browser
2. Open the browser's Developer Tools (F12)
3. Go to the **Network** tab
4. Look for requests to `/_vercel/insights/` or `/script.js`
5. You should see a successful request (200 status)

Alternatively, check the application logs for the message:
```
Vercel Analytics enabled: auto (from Vercel deployment)
```

## Deployment on Vercel

When deploying to Vercel, the analytics configuration is simplified:

### Option 1: Deploy with Vercel CLI

```bash
# Install Vercel CLI globally
npm i -g vercel

# Deploy to Vercel
vercel

# Deploy to production
vercel --prod
```

### Option 2: Deploy via Git Integration

1. Push your code to GitHub, GitLab, or Bitbucket
2. Import your repository in the Vercel Dashboard
3. Vercel will automatically detect and deploy your project

### Environment Variables on Vercel

Set the environment variable in your Vercel project settings:

1. Go to your project in Vercel Dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add:
   - Key: `VERCEL_ANALYTICS_ENABLED`
   - Value: `true`
   - Environment: Production, Preview, Development (as needed)

## How It Works

The integration uses Vercel's generic analytics approach suitable for any web framework:

1. **Analytics Module** (`utils/vercel_analytics.py`):
   - Provides helper functions to generate analytics scripts
   - Reads configuration from environment variables
   - Returns JavaScript code to inject into Gradio

2. **Dashboard Integration** (`ui/pipeline_dashboard.py`):
   - Loads environment variables using `python-dotenv`
   - Calls analytics helper functions
   - Injects scripts via Gradio's `js` and `head` parameters

3. **Script Injection**:
   - The analytics script is added to Gradio's `launch()` method
   - It initializes the Vercel Analytics client
   - Automatically tracks page views and interactions

## Configuration Options

### VERCEL_ANALYTICS_ENABLED

- **Type**: Boolean (as string)
- **Default**: `false`
- **Values**: `true` or `false`
- **Description**: Enable or disable Vercel Analytics

### VERCEL_ANALYTICS_ID (Optional)

- **Type**: String
- **Default**: Auto-configured by Vercel
- **Description**: Custom analytics ID (rarely needed)

## Viewing Analytics Data

After your dashboard has some traffic:

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Select your project
3. Navigate to the **Analytics** tab
4. View metrics including:
   - Page views
   - Unique visitors
   - Top pages
   - Traffic sources
   - Device/browser breakdown

## Troubleshooting

### Analytics not tracking

1. **Check environment variable**: Ensure `VERCEL_ANALYTICS_ENABLED=true`
2. **Restart the dashboard**: Environment changes require a restart
3. **Check browser console**: Look for JavaScript errors
4. **Verify Vercel deployment**: Analytics only works on Vercel-deployed apps or with Vercel CLI

### Local Development

For local development, analytics won't work unless you're using Vercel CLI:

```bash
# Run locally with Vercel CLI
vercel dev
```

This sets up the proper Vercel environment including analytics routes.

### Script not loading

If the analytics script isn't loading:

1. Check that you've enabled analytics in Vercel Dashboard
2. Verify the `/_vercel/insights/script.js` route is accessible
3. Check for Content Security Policy (CSP) issues in browser console
4. Ensure your deployment is complete and active

## Privacy and Compliance

Vercel Web Analytics is privacy-friendly:

- No cookies used
- No personal data collected
- GDPR and CCPA compliant
- Data is aggregated and anonymized

For more information, see [Vercel's Privacy Policy](https://vercel.com/legal/privacy-policy).

## Additional Resources

- [Vercel Web Analytics Documentation](https://vercel.com/docs/analytics)
- [Vercel Analytics Quickstart](https://vercel.com/docs/analytics/quickstart)
- [Gradio Documentation](https://gradio.app/docs)
- [python-dotenv Documentation](https://github.com/theskumar/python-dotenv)

## Support

For issues related to:

- **Analytics integration**: Check this repository's issues
- **Vercel platform**: Contact [Vercel Support](https://vercel.com/support)
- **Gradio framework**: See [Gradio Documentation](https://gradio.app)
