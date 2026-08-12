# compose-bench report

Model: `x-ai/grok-4.5`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 3 | 23.1±2.3 | 4.3±0.5 | 3.3±0.5 | 60% (3/3 runs) | 1.0 | 16923.7±1469.7 | 5423.7±208.8 | 0.0171±0.0018 | 1.00±0.00 |
| calendar-scheduling | runlet | 3 | 28.6±5.6 | 3.7±0.5 | 3.0 | 67% (3/3 runs) | 1.0 | 21038.0±2293.3 | 7783.0±448.4 | 0.0221±0.0035 | 1.00±0.00 |
| crm-hygiene | compose | 3 | 17.0±1.1 | 4.0 | 3.0 | 67% (3/3 runs) | 1.0 | 15728.7±904.9 | 5069.7±149.5 | 0.0142±0.0003 | 1.00±0.00 |
| crm-hygiene | runlet | 3 | 18.0±3.5 | 3.0±0.8 | 2.0±0.8 | 100% (3/3 runs) | 1.3±0.5 | 17642.7±5982.6 | 6967.3±1081.7 | 0.0179±0.0057 | 0.72±0.39 |
| log-incident | compose | 3 | 16.1±3.4 | 5.0±0.8 | 4.0±0.8 | 75% (3/3 runs) | 1.3±0.5 | 27436.7±6715.0 | 8937.0±1102.0 | 0.0169±0.0039 | 1.00±0.00 |
| log-incident | runlet | 3 | 14.3±1.3 | 5.0 | 4.0 | 75% (3/3 runs) | 1.0 | 33309.7±4351.1 | 10215.7±1500.4 | 0.0190±0.0010 | 1.00±0.00 |
| revenue-report | compose | 3 | 10.3±0.4 | 4.0 | 3.0 | 67% (3/3 runs) | 1.0 | 12449.0±443.4 | 3796.0±101.4 | 0.0083±0.0012 | 1.00±0.00 |
| revenue-report | runlet | 3 | 14.2±2.5 | 3.3±0.5 | 2.3±0.5 | 100% (3/3 runs) | 1.0 | 18166.0±2991.0 | 6395.7±301.5 | 0.0134±0.0028 | 1.00±0.00 |
| support-triage | compose | 3 | 23.7±16.6 | 10.0±8.5 | 9.0±8.5 | 19% (3/3 runs) | 1.0 | 57070.3±60234.3 | 5749.3±2031.5 | 0.0201±0.0158 | 1.00±0.00 |
| support-triage | runlet | 3 | 9.7±1.3 | 3.7±0.5 | 2.7±0.5 | 62% (3/3 runs) | 0.7±0.5 | 16351.7±3589.2 | 6101.0±194.3 | 0.0169±0.0043 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
