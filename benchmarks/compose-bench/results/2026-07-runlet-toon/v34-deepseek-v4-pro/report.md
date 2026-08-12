# compose-bench report

Model: `deepseek/deepseek-v4-pro`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 3 | 66.9±5.7 | 5.7±0.5 | 6.7±2.4 | 35% (3/3 runs) | 1.0 | 27130.7±7451.3 | 8677.0±2139.0 | 0.0082±0.0016 | 0.67±0.47 |
| calendar-scheduling | runlet | 3 | 99.5±44.6 | 7.0±2.2 | 12.3±10.5 | 30% (3/3 runs) | 3.0±1.4 | 71286.7±39674.5 | 16292.7±7010.9 | 0.0142±0.0067 | 1.00±0.00 |
| crm-hygiene | compose | 3 | 69.3±4.3 | 5.7±0.5 | 6.0±1.4 | 33% (3/3 runs) | 0.7±0.5 | 34025.3±989.3 | 11131.0±575.4 | 0.0096±0.0005 | 1.00±0.00 |
| crm-hygiene | runlet | 3 | 122.2±38.5 | 6.3±0.9 | 6.7±0.9 | 45% (3/3 runs) | 1.3±0.9 | 63421.7±18296.6 | 17773.3±4973.7 | 0.0165±0.0050 | 0.93±0.09 |
| log-incident | compose | 3 | 36.7±18.1 | 4.7±2.1 | 7.0±4.2 | 5% (1/3 runs) | 0.3±0.5 | 26208.7±15622.3 | 7732.7±3258.4 | 0.0051±0.0025 | 0.67±0.47 |
| log-incident | runlet | 3 | 45.6±5.4 | 5.7±0.5 | 10.3±3.8 | 10% (1/3 runs) | 0.3±0.5 | 48965.0±3971.0 | 13301.3±495.2 | 0.0078±0.0008 | 1.00±0.00 |
| revenue-report | compose | 3 | 29.5±7.6 | 5.0±0.8 | 4.7±1.7 | 36% (3/3 runs) | 0.7±0.5 | 19286.7±5473.9 | 5132.7±1055.2 | 0.0037±0.0011 | 1.00±0.00 |
| revenue-report | runlet | 3 | 44.0±16.9 | 5.0±0.8 | 4.7±1.7 | 36% (3/3 runs) | 0.7±0.5 | 30364.0±7678.7 | 8123.0±1794.2 | 0.0062±0.0026 | 1.00±0.00 |
| support-triage | compose | 3 | 24.1±21.8 | 3.3±1.9 | 2.3±1.9 | 57% (1/3 runs) | 0.3±0.5 | 14503.3±14017.2 | 4835.0±2800.2 | 0.0032±0.0026 | 0.33±0.47 |
| support-triage | runlet | 3 | 34.8±19.9 | 4.7±2.1 | 8.7±8.8 | 15% (2/3 runs) | 1.0±0.8 | 34235.3±20851.8 | 8972.3±3539.9 | 0.0058±0.0036 | 0.67±0.47 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
