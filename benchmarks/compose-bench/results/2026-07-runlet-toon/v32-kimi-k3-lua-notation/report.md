# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 3 | 54.6±3.0 | 4.0 | 3.0 | 67% (3/3 runs) | 1.0 | 12576.0±113.4 | 4888.0±305.7 | 0.0328±0.0034 | 1.00±0.00 |
| crm-hygiene | compose | 3 | 80.1±9.5 | 5.0±0.8 | 4.7±1.7 | 43% (3/3 runs) | 0.7±0.5 | 24914.0±1143.2 | 8460.0±777.2 | 0.0526±0.0084 | 1.00±0.00 |
| log-incident | compose | 3 | 67.6±12.9 | 6.0±1.4 | 5.3±1.9 | 62% (3/3 runs) | 1.0 | 26915.3±8083.2 | 7281.0±784.5 | 0.0371±0.0061 | 1.00±0.00 |
| revenue-report | compose | 3 | 45.6±10.6 | 4.3±0.5 | 3.3±0.5 | 70% (3/3 runs) | 1.3±0.5 | 13343.3±2784.7 | 4233.7±591.8 | 0.0244±0.0063 | 1.00±0.00 |
| support-triage | compose | 3 | 64.4±25.7 | 4.7±0.9 | 3.7±0.9 | 73% (3/3 runs) | 1.3±0.5 | 19034.3±9258.6 | 5539.0±1587.2 | 0.0358±0.0134 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
