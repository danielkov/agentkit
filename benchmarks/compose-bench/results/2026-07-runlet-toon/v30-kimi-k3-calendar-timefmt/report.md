# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | runlet | 3 | 108.0±26.9 | 3.7±0.9 | 2.7±0.9 | 100% (3/3 runs) | 1.7±0.9 | 29652.0±14168.4 | 10506.7±2267.6 | 0.0745±0.0222 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
