# Security Guidelines

This document outlines security best practices for the frontend application.

---

## Environment Variables

### Frontend (VITE\_\* prefix)

**Important**: Only `VITE_*` prefixed variables are accessible in client-side code. These variables are **BAKED INTO THE BUILD** at compile time and become part of the JavaScript bundle that is sent to users' browsers.

### Safe to expose (anon key):

- `VITE_SUPABASE_URL` - Public Supabase project URL
- `VITE_SUPABASE_PUBLISHABLE_KEY` - Anon key (protected by RLS policies)
- `VITE_DEMO_MODE` - Boolean flag to enable demo mode
- `VITE_DEMO_EMAIL`, `VITE_DEMO_PASSWORD`, `VITE_DEMO_USERNAME` - Demo credentials (only for development/demo environments)

**Note**: The Supabase "anon" key is safe to expose because all data access is protected by Row Level Security (RLS) policies on the database.

### NEVER expose in frontend:

- ❌ Service role keys
- ❌ API secrets
- ❌ Database passwords
- ❌ Private keys
- ❌ JWT signing secrets

### Environment File Management:

- **DO commit**: `.env.example` (with placeholder values)
- **DO NOT commit**: `.env`, `.env.local`, `.env.production`, or any file with actual credentials
- **Production deployments**: Use Azure Application Settings or GitHub Secrets to inject environment variables at deployment time

---

## Authentication

### Token Storage

**Current Implementation**:

- Auth tokens are stored in `localStorage` (Supabase default behavior)
- Session tokens are automatically refreshed by Supabase client
- "Remember me" functionality toggles between `localStorage` and `sessionStorage`

**Security Considerations**:

- `localStorage` is vulnerable to XSS (Cross-Site Scripting) attacks
- Ensure no XSS vulnerabilities exist in the application
- All user input is rendered as text (never as HTML)
- No use of `dangerouslySetInnerHTML` or `innerHTML`

**Alternative** (requires backend changes):

- Use `httpOnly` cookies for token storage
- Requires custom auth endpoints to set cookies
- Provides protection against XSS token theft

### Session Management

- Sessions are managed by Supabase Auth
- Access tokens expire after 1 hour (default)
- Refresh tokens are automatically used to get new access tokens
- Users can manually log out to invalidate tokens

---

## Authorization

### Client-Side Role Checks

⚠️ **CRITICAL**: Frontend role checks are **UI-ONLY** and provide no security.

**All authorization MUST be validated on the backend via**:

1. Supabase Row Level Security (RLS) policies
2. Edge function permission checks
3. Database triggers

### Admin Routes

- The `AdminRoute` component gates admin-only pages in the UI
- **Backend must validate user role on every API call**
- **Never trust client-provided role values**

Example of proper backend validation:

```typescript
// In Supabase Edge Function
const authHeader = req.headers.get("Authorization");
if (!authHeader) return new Response("Unauthorized", { status: 401 });

const token = authHeader.replace("Bearer ", "");
const {
  data: { user },
} = await supabase.auth.getUser(token);
if (!user) return new Response("Unauthorized", { status: 401 });

// Fetch user profile and check role
const { data: profile } = await supabase
  .from("profiles")
  .select("role")
  .eq("id", user.id)
  .single();

if (profile?.role !== "admin") {
  return new Response("Forbidden", { status: 403 });
}
```

---

## Row Level Security (RLS)

All Supabase tables **MUST** have RLS policies enabled. Below are the recommended policies:

### `profiles` table

```sql
-- Users can read their own profile
CREATE POLICY "Users can read own profile"
  ON profiles FOR SELECT
  USING (auth.uid() = id);

-- Admins can read all profiles
CREATE POLICY "Admins can read all profiles"
  ON profiles FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );

-- Users can update their own profile (except role)
CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = id)
  WITH CHECK (
    auth.uid() = id AND
    (SELECT role FROM profiles WHERE id = auth.uid()) = role -- Role cannot be changed
  );
```

### `buildings` table

```sql
-- Users can only access their organization's buildings
CREATE POLICY "Users can access org buildings"
  ON buildings FOR SELECT
  USING (
    organization_id IN (
      SELECT organization_id FROM profiles WHERE id = auth.uid()
    )
  );
```

### `appliances` table

```sql
-- Users can only access appliances in their buildings
CREATE POLICY "Users can access org appliances"
  ON appliances FOR SELECT
  USING (
    building_id IN (
      SELECT id FROM buildings WHERE organization_id IN (
        SELECT organization_id FROM profiles WHERE id = auth.uid()
      )
    )
  );
```

### `model_versions` table

```sql
-- Restricted to model owners and admins
CREATE POLICY "Model owners can access versions"
  ON model_versions FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM appliance_models
      WHERE id = model_id AND (
        created_by = auth.uid() OR
        EXISTS (SELECT 1 FROM profiles WHERE id = auth.uid() AND role = 'admin')
      )
    )
  );
```

---

## Edge Functions

All Supabase Edge Functions **MUST** follow these security practices:

### 1. Verify JWT Token

```typescript
const authHeader = req.headers.get("Authorization");
if (!authHeader) {
  return new Response("Unauthorized", { status: 401 });
}

const token = authHeader.replace("Bearer ", "");
const {
  data: { user },
  error,
} = await supabase.auth.getUser(token);

if (error || !user) {
  return new Response("Unauthorized", { status: 401 });
}
```

### 2. Extract User ID from Token

```typescript
// Use user.id for all authorization checks
const userId = user.id;
```

### 3. Validate Permissions

```typescript
// Check user has permission for this operation
const { data: profile } = await supabase
  .from("profiles")
  .select("role, organization_id")
  .eq("id", userId)
  .single();

if (!profile) {
  return new Response("Forbidden", { status: 403 });
}

// For admin-only operations
if (profile.role !== "admin") {
  return new Response("Forbidden - Admin required", { status: 403 });
}
```

### 4. Never Trust Client Input

```typescript
// ❌ BAD: Trust client-provided user ID
const requestUserId = await req.json().userId;
// ... use requestUserId

// ✅ GOOD: Use authenticated user ID from token
const authenticatedUserId = user.id;
// ... use authenticatedUserId
```

---

## Console Logging

### Production Builds

- Production builds automatically strip all `console.*` calls via esbuild
- Configuration in `vite.config.ts`:
  ```typescript
  esbuild: {
    drop: mode === 'production' ? ['console', 'debugger'] : [],
  }
  ```

### Development Logging

**Never log sensitive data**:

- ❌ Auth tokens
- ❌ Passwords
- ❌ API keys
- ❌ Personally Identifiable Information (PII)
- ❌ Complete user objects with email

**Use environment checks for debug logging**:

```typescript
if (import.meta.env.DEV) {
  console.log("Debug info:", data);
}
```

---

## Dependency Security

### Regular Updates

- Run `npm audit` regularly to identify vulnerabilities
- Update dependencies with security patches promptly
- Review changelogs for breaking changes before updating

### Audit Command

```bash
cd frontend
npm audit

# Fix non-breaking vulnerabilities automatically
npm audit fix

# For breaking changes, review and update manually
npm audit fix --force  # Use with caution
```

### Dependency Review Process

1. Check audit report for severity (critical, high, moderate, low)
2. Review CVE details for exploitability in this codebase
3. Update affected packages
4. Test thoroughly after updates
5. Commit dependency updates separately from feature changes

---

## Common Vulnerabilities & Mitigations

### XSS (Cross-Site Scripting)

**Prevention**:

- ✅ All user input is rendered as text (React default)
- ✅ No use of `dangerouslySetInnerHTML`
- ✅ No direct DOM manipulation with `innerHTML`
- ✅ No `eval()` or `new Function()` with user input

**If HTML rendering is required**:

```typescript
import DOMPurify from "dompurify";

const cleanHtml = DOMPurify.sanitize(userInput);
```

### CSRF (Cross-Site Request Forgery)

**Mitigation**:

- Supabase JWT tokens are sent in `Authorization` header (not cookies)
- SameSite cookie flags for any custom cookies
- API endpoints validate JWT on every request

### SQL Injection

**Prevention**:

- Use Supabase client library (parameterized queries by default)
- Never construct SQL strings from user input
- RLS policies protect against unauthorized data access

### Authentication Bypass

**Prevention**:

- All API routes validate JWT token
- Frontend route guards are UI-only (backend validation required)
- RLS policies enforce database-level access control

---

## Security Checklist

Before deploying to production, verify:

- [ ] All environment files (`.env`, `.env.production`) are in `.gitignore`
- [ ] No secrets committed to git history
- [ ] All Supabase tables have RLS policies enabled
- [ ] Edge functions validate JWT tokens
- [ ] Edge functions check user permissions
- [ ] Admin routes use `AdminRoute` component
- [ ] Console logs are stripped in production builds
- [ ] Dependencies have no critical vulnerabilities (`npm audit`)
- [ ] HTTPS is enforced on production domain
- [ ] Supabase anon key is the only key in frontend
- [ ] Service role key is never used in frontend

---

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** open a public GitHub issue
2. Email security concerns to the project maintainer
3. Include details: affected component, reproduction steps, potential impact
4. Allow time for a fix before public disclosure

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Supabase Security Best Practices](https://supabase.com/docs/guides/auth/row-level-security)
- [Content Security Policy (CSP)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [React Security Best Practices](https://react.dev/learn/sharing-state-between-components#security)
