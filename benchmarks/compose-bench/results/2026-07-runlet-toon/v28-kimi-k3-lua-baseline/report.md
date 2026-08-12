# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 3 | 78.8±15.3 | 4.0 | 3.0 | 67% (3/3 runs) | 1.0 | 14266.0±295.3 | 5150.3±247.0 | 0.0355±0.0034 | 1.00±0.00 |
| crm-hygiene | compose | 3 | 125.9±42.6 | 5.7±0.5 | 5.0±0.8 | 60% (3/3 runs) | 0.7±0.5 | 31758.7±3539.5 | 8898.3±1532.3 | 0.0686±0.0236 | 1.00±0.00 |
| log-incident | compose | 3 | 82.3±6.3 | 8.3±0.5 | 8.0±1.4 | 67% (3/3 runs) | 2.3±0.5 | 39232.7±2293.3 | 7878.0±580.8 | 0.0450±0.0017 | 1.00±0.00 |
| revenue-report | compose | 3 | 59.7±11.9 | 4.0 | 3.0 | 67% (3/3 runs) | 1.0 | 13046.3±409.5 | 4379.0±173.6 | 0.0265±0.0012 | 1.00±0.00 |
| support-triage | compose | 3 | 90.5±17.9 | 4.3±0.5 | 3.3±0.5 | 70% (3/3 runs) | 1.0 | 21096.0±5604.2 | 7272.3±1813.2 | 0.0441±0.0084 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
