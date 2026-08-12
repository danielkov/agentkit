# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | runlet | 3 | 140.0±13.1 | 2.7±0.5 | 1.7±0.5 | 100% (3/3 runs) | 0.7±0.5 | 21697.0±7186.8 | 10276.0±2496.0 | 0.0744±0.0164 | 1.00±0.00 |
| log-incident | runlet | 3 | 119.3±41.6 | 4.0±0.8 | 3.0±0.8 | 78% (3/3 runs) | 1.0 | 31167.3±3245.9 | 11501.3±1725.3 | 0.0656±0.0167 | 0.89±0.16 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
