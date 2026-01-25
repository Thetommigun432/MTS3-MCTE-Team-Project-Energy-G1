
import { test, expect } from '@playwright/test';

test('Dashboard loads and shows predictions', async ({ page }) => {
    // 1. Visit Dashboard
    await page.goto('/');

    // 2. Check Title or Header
    await expect(page).toHaveTitle(/Energy|NILM/i);

    // 3. Check for Predictions Table or Chart
    // Assuming there's a component with a specific ID or class
    // We'll look for generic text first since we don't know exact selectors yet
    await expect(page.getByText(/Predictions/i)).toBeVisible();

    // 4. Check for presence of "HeatPump" (from our dummy model)
    // This asserts that data flowed from Producer -> Redis -> Service -> Influx -> Frontend
    // Note: This might be flaky if the frontend doesn't auto-refresh or data is delayed.
    // We'll give it a generous timeout if possible.

    // Wait for data
    await expect(page.getByText('HeatPump')).toBeVisible({ timeout: 15000 });
});

test('Models page lists available models', async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByText(/Transformer|WaveNILM|Dummy/i)).toBeVisible();
});
