# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | runlet | 3 | 124.4±12.7 | 3.0 | 2.0 | 100% (3/3 runs) | 1.0 | 25548.7±1363.5 | 10864.0±691.9 | 0.0691±0.0066 | 1.00±0.00 |
| crm-hygiene | runlet | 3 | 210.5±37.8 | 2.7±0.5 | 1.7±0.5 | 100% (3/3 runs) | 0.7±0.5 | 29280.0±6680.5 | 14421.3±2628.0 | 0.1366±0.0317 | 1.00±0.00 |
| log-incident | runlet | 3 | 155.5±38.7 | 4.7±1.2 | 3.7±1.2 | 82% (3/3 runs) | 1.0±0.8 | 41906.0±6023.8 | 13807.0±1233.5 | 0.0807±0.0211 | 1.00±0.00 |
| revenue-report | runlet | 3 | 102.1±49.3 | 2.7±0.9 | 1.7±0.9 | 100% (3/3 runs) | 0.7±0.9 | 20453.3±14033.4 | 9029.3±3723.8 | 0.0635±0.0268 | 1.00±0.00 |
| support-triage | runlet | 3 | 55.0±10.4 | 2.0 | 1.0 | 100% (3/3 runs) | 0.0 | 9848.0±990.2 | 6089.0±511.4 | 0.0386±0.0091 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
