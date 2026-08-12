# compose-bench report

Model: `x-ai/grok-4.5`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| crm-hygiene | runlet | 8 | 26.9±5.9 | 4.0±0.5 | 3.0±0.5 | 100% (8/8 runs) | 2.0±0.5 | 25919.0±5903.2 | 8262.0±1918.5 | 0.0275±0.0082 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
