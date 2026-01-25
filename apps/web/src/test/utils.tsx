/**
 * Test utilities for React component testing.
 * 
 * Provides wrappers and helpers for common testing patterns.
 */
import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

/**
 * Custom render function that wraps components with common providers.
 * Use this instead of RTL's render() for components that need routing.
 */
interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
    route?: string;
}

function AllProviders({ children }: { children: React.ReactNode }) {
    return (
        <BrowserRouter>
            {children}
        </BrowserRouter>
    );
}

export function renderWithProviders(
    ui: ReactElement,
    options?: CustomRenderOptions
) {
    if (options?.route) {
        window.history.pushState({}, 'Test page', options.route);
    }

    return render(ui, { wrapper: AllProviders, ...options });
}

// Re-export everything from RTL for convenience
export * from '@testing-library/react';
export { renderWithProviders as render };
