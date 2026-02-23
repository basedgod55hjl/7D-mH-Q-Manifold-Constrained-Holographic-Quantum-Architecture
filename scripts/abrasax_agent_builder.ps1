# ABRASAX GOD OS - Auto Agent Builder
# Loads LM Studio model and builds agents via /v1/responses API
# Sir Charles Spikes | Cincinnati, Ohio, USA

param(
    [string]$Model = "qwen/qwen3-vl-4b",
    [string]$Input = "What is the weather like in Boston today?",
    [string]$BaseUrl = "http://localhost:1234",
    [switch]$UseBridge
)

$bridgeUrl = "http://localhost:17777"
$targetUrl = if ($UseBridge) { "$bridgeUrl/v1/responses" } else { "$BaseUrl/v1/responses" }

$headers = @{
    "Content-Type" = "application/json"
}
if ($env:LM_STUDIO_API_TOKEN) {
    $headers["Authorization"] = "Bearer $env:LM_STUDIO_API_TOKEN"
} elseif ($env:LM_API_TOKEN) {
    $headers["Authorization"] = "Bearer $env:LM_API_TOKEN"
}

$body = @{
    model = $Model
    input = $Input
    tools = @(
        @{
            type = "function"
            name = "get_current_weather"
            description = "Get the current weather in a given location"
            parameters = @{
                type = "object"
                properties = @{
                    location = @{
                        type = "string"
                        description = "The city and state, e.g. San Francisco, CA"
                    }
                    unit = @{
                        type = "string"
                        enum = @("celsius", "fahrenheit")
                    }
                }
                required = @("location", "unit")
            }
        }
    )
    tool_choice = "auto"
} | ConvertTo-Json -Depth 10

Write-Host "ABRASAX Agent Builder - Calling LM Studio /v1/responses" -ForegroundColor Cyan
Write-Host "Target: $targetUrl" -ForegroundColor Gray
Write-Host "Model:  $Model" -ForegroundColor Gray

try {
    $response = Invoke-RestMethod -Uri $targetUrl -Method Post -Headers $headers -Body $body
    $response | ConvertTo-Json -Depth 20
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Ensure LM Studio is running on port 1234 (or 17777 for bridge)" -ForegroundColor Yellow
    exit 1
}
