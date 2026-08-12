# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | runlet | 3 | 180.9±69.6 | 5.0±2.2 | 4.0±2.2 | 100% (3/3 runs) | 2.7±1.7 | 54255.7±33555.6 | 14725.0±4429.6 | 0.1124±0.0446 | 0.00±0.00 |
| crm-hygiene | runlet | 3 | 118.9±33.8 | 2.3±0.5 | 1.3±0.5 | 100% (3/3 runs) | 0.3±0.5 | 15566.3±5419.0 | 8587.0±2233.9 | 0.0668±0.0151 | 1.00±0.00 |
| log-incident | runlet | 3 | 63.5±18.4 | 4.3±0.9 | 4.3±2.1 | 46% (3/3 runs) | 0.7±0.5 | 29347.0±5663.3 | 9552.7±853.5 | 0.0421±0.0083 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
