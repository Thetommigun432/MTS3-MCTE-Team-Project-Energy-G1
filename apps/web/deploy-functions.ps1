# PowerShell script to deploy all Supabase Edge Functions
# Usage: .\deploy-functions.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deploying Supabase Edge Functions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to frontend directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if we're in the right directory
if (-not (Test-Path "supabase/functions")) {
    Write-Host "ERROR: supabase/functions directory not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this from the frontend directory." -ForegroundColor Red
    exit 1
}

# List of all edge functions to deploy
$functions = @(
    "log-login-event",
    "log-auth-event",
    "invite-user",
    "admin-invite",
    "accept-invite",
    "delete-account",
    "get-readings",
    "get-dashboard-data",
    "generate-report",
    "upsert-avatar",
    "run-inference",
    "register-model",
    "create-model-version-upload",
    "finalize-model-version",
    "set-active-model-version",
    "parse-nilm-csv",
    "ensure-demo-user"
)

$successCount = 0
$failCount = 0
$failedFunctions = @()

Write-Host "Found $($functions.Count) functions to deploy" -ForegroundColor Yellow
Write-Host ""

foreach ($func in $functions) {
    Write-Host "[$($successCount + $failCount + 1)/$($functions.Count)] Deploying: $func" -ForegroundColor Green

    # Check if function directory exists
    if (-not (Test-Path "supabase/functions/$func")) {
        Write-Host "  ⚠️  Function directory not found, skipping..." -ForegroundColor Yellow
        Write-Host ""
        continue
    }

    # Deploy the function using npx
    try {
        $output = npx supabase functions deploy $func 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Success!" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "  ✗ Failed!" -ForegroundColor Red
            Write-Host "  Error: $output" -ForegroundColor Red
            $failCount++
            $failedFunctions += $func
        }
    } catch {
        Write-Host "  ✗ Failed with exception!" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
        $failCount++
        $failedFunctions += $func
    }

    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Deployment Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total functions: $($functions.Count)" -ForegroundColor White
Write-Host "✓ Successful: $successCount" -ForegroundColor Green
Write-Host "✗ Failed: $failCount" -ForegroundColor Red

if ($failCount -gt 0) {
    Write-Host ""
    Write-Host "Failed functions:" -ForegroundColor Red
    foreach ($func in $failedFunctions) {
        Write-Host "  - $func" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "To retry failed functions, run:" -ForegroundColor Yellow
    foreach ($func in $failedFunctions) {
        Write-Host "  npx supabase functions deploy $func" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Set environment secrets:" -ForegroundColor White
Write-Host "   npx supabase secrets set SITE_URL=`"https://energymonitorstorage.z1.web.core.windows.net`"" -ForegroundColor Gray
Write-Host ""
Write-Host "2. View function logs:" -ForegroundColor White
Write-Host "   npx supabase functions logs <function-name>" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test a function:" -ForegroundColor White
Write-Host "   curl -X POST 'https://bhdcbvruzvhmcogxfkil.supabase.co/functions/v1/<function-name>' \" -ForegroundColor Gray
Write-Host "     -H 'Authorization: Bearer <token>' \" -ForegroundColor Gray
Write-Host "     -H 'Content-Type: application/json'" -ForegroundColor Gray
Write-Host ""

if ($successCount -eq $functions.Count) {
    Write-Host "✨ All functions deployed successfully! ✨" -ForegroundColor Green
    exit 0
} else {
    Write-Host "⚠️  Some functions failed to deploy. Check errors above." -ForegroundColor Yellow
    exit 1
}
