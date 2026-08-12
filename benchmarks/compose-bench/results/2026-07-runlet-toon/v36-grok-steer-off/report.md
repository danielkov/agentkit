# compose-bench report

Model: `x-ai/grok-4.5`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| crm-hygiene | runlet | 8 | 25.1±7.2 | 5.1±1.2 | 4.1±1.2 | 76% (8/8 runs) | 1.8±0.7 | 38447.6±17311.0 | 9627.6±2536.1 | 0.0260±0.0078 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
