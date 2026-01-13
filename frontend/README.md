# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Demo Mode

This project includes a demo mode for presentations and testing. When enabled, it shows demo credentials on the login page and allows quick login as a demo admin user.

### Enabling Demo Mode

1. Set the environment variable in your `.env` file:
   ```
   VITE_DEMO_MODE="true"
   ```

2. Set the Supabase Edge Function secret (in Supabase Dashboard > Edge Functions > Secrets):
   ```
   DEMO_MODE_ENABLED=true
   ```

### Demo Credentials

When demo mode is enabled:
- **Username:** `admin`
- **Password:** `admin123`
- **Email:** `admin@demo.local`

You can either:
- Click the "Log in as demo admin" button on the login page
- Type `admin` in the email field (it will auto-map to `admin@demo.local`)
- Use the full email `admin@demo.local` with password `admin123`

### Disabling Demo Mode

For production deployments, ensure demo mode is disabled:
- Remove or set `VITE_DEMO_MODE="false"` in your environment
- Remove or set `DEMO_MODE_ENABLED=false` in Supabase Edge Function secrets

This ensures:
- Demo credentials UI is hidden
- Username shortcuts are disabled
- The `ensure-demo-user` edge function will reject requests

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
