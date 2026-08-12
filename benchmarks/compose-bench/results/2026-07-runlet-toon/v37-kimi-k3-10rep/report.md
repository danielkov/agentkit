# compose-bench report

Model: `moonshotai/kimi-k3`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 10 | 68.4±21.1 | 4.5±0.7 | 3.5±0.7 | 71% (10/10 runs) | 1.2±0.4 | 18997.1±8075.2 | 6460.9±1968.1 | 0.0497±0.0153 | 1.00±0.00 |
| calendar-scheduling | runlet | 10 | 143.3±46.3 | 3.1±1.4 | 2.9±3.8 | 55% (10/10 runs) | 0.6±0.7 | 28686.1±11949.5 | 11854.0±2869.4 | 0.0974±0.0276 | 1.00±0.00 |
| crm-hygiene | compose | 10 | 66.6±12.5 | 4.5±0.7 | 3.9±1.2 | 51% (10/10 runs) | 0.7±0.5 | 22541.9±5937.8 | 7936.7±1960.0 | 0.0487±0.0118 | 1.00±0.00 |
| crm-hygiene | runlet | 10 | 131.2±66.1 | 2.4±0.7 | 1.6±1.0 | 75% (10/10 runs) | 0.2±0.4 | 19981.7±8479.8 | 9394.3±2443.6 | 0.0951±0.0520 | 1.00±0.00 |
| log-incident | compose | 10 | 50.9±7.7 | 5.3±0.5 | 5.0±0.9 | 54% (10/10 runs) | 1.1±0.3 | 23175.7±2983.4 | 7305.3±356.6 | 0.0364±0.0052 | 1.00±0.00 |
| log-incident | runlet | 10 | 136.3±135.9 | 3.9±1.3 | 2.9±1.3 | 69% (10/10 runs) | 0.4±0.5 | 36391.9±17363.3 | 14257.4±9019.2 | 0.0945±0.0847 | 1.00±0.00 |
| revenue-report | compose | 10 | 47.5±16.5 | 4.1±0.3 | 3.1±0.3 | 68% (10/10 runs) | 1.1±0.3 | 14455.2±3991.7 | 4610.5±1102.6 | 0.0280±0.0095 | 1.00±0.00 |
| revenue-report | runlet | 10 | 57.5±33.8 | 2.1±0.3 | 1.1±0.3 | 100% (10/10 runs) | 0.1±0.3 | 12545.3±5271.5 | 6773.0±2147.3 | 0.0440±0.0264 | 1.00±0.00 |
| support-triage | compose | 10 | 66.3±26.0 | 4.8±1.2 | 7.3±7.1 | 26% (10/10 runs) | 1.1±0.3 | 23983.8±13481.5 | 7212.8±2680.4 | 0.0442±0.0172 | 1.00±0.00 |
| support-triage | runlet | 10 | 47.5±18.2 | 2.1±0.3 | 1.1±0.3 | 91% (10/10 runs) | 0.0 | 12460.9±4021.6 | 6692.4±1479.8 | 0.0353±0.0155 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
