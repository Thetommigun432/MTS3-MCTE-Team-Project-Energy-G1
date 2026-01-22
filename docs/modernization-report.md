# Frontend Modernization Report

## 1. Current Status (Baseline)

### Toolchain Versions
- **Node**: `v24.11.1` (Local), Cloudflare uses 20 (from .nvmrc)
- **Vite**: `7.3.1` (Modern)
- **React**: `19.2.3` (Modern)
- **TypeScript**: `5.9.3`
- **Vitest**: `1.6.1` (Outdated, current is 3.x)

### Dependency Audit
- **Moderate Vulnerabilities**: 5 (from `npm audit`)
- **Outdated Packages**:
  - `vitest`: 1.6.1 -> 3.x (Major)
  - `@testing-library/react`: Check version
  - `eslint`: Check version

## 2. Risk Assessment

### High Risk
- **Vitest 1.x -> 3.x**: Major breaking changes in test APIs/configuration. Requires careful verification of test suite.
- **Node Target -> 22**: Generally safe, but ensures long-term support.

### Low Risk
- **Vite**: Already on v7.
- **React**: Already on v19.

## 3. Modernization Plan

1.  **Node.js**: Pin to `22` (Active LTS) in `.nvmrc` and `engines`.
2.  **Vitest**: Upgrade to `3.x` to match Vite 7.
3.  **Linting**: Ensure ESLint 9+ with Flat Config.
4.  **Security**: Resolve 5 moderate vulnerabilities.
5.  **Deprecations**: Scan and fix.

## 4. Cloudflare Config
- **Root**: `/`
- **Build**: `npm ci && npm run build`
- **Output**: `apps/web/dist`
