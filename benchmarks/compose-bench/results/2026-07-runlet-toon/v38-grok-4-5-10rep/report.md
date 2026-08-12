# compose-bench report

Model: `x-ai/grok-4.5`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 10 | 20.1±3.3 | 3.7±0.9 | 2.7±0.9 | 67% (9/10 runs) | 0.9±0.3 | 15388.3±4373.2 | 5145.5±944.7 | 0.0145±0.0050 | 0.90±0.30 |
| calendar-scheduling | runlet | 10 | 33.2±9.1 | 4.0±0.9 | 3.0±0.9 | 87% (10/10 runs) | 0.9±0.5 | 30335.4±12156.8 | 10113.2±2590.0 | 0.0305±0.0090 | 1.00±0.00 |
| crm-hygiene | compose | 10 | 17.0±1.2 | 4.0 | 3.0 | 67% (10/10 runs) | 1.0 | 16060.0±902.5 | 5114.9±587.5 | 0.0136±0.0015 | 1.00±0.00 |
| crm-hygiene | runlet | 10 | 22.3±5.3 | 3.7±1.1 | 2.8±1.0 | 100% (10/10 runs) | 1.3±0.8 | 27428.2±12321.0 | 9038.3±2512.1 | 0.0226±0.0068 | 1.00±0.00 |
| log-incident | compose | 10 | 14.2±2.3 | 4.8±0.7 | 4.8±2.6 | 48% (10/10 runs) | 1.0 | 22435.1±4841.2 | 7151.8±1134.9 | 0.0143±0.0029 | 1.00±0.00 |
| log-incident | runlet | 10 | 12.7±3.2 | 3.7±0.8 | 2.7±0.8 | 78% (10/10 runs) | 0.3±0.5 | 26506.3±6432.5 | 9744.8±1200.6 | 0.0151±0.0044 | 1.00±0.00 |
| revenue-report | compose | 10 | 9.4±0.5 | 4.0 | 3.0 | 67% (10/10 runs) | 1.0 | 12478.9±724.7 | 3646.8±160.8 | 0.0076±0.0012 | 1.00±0.00 |
| revenue-report | runlet | 10 | 14.6±5.4 | 3.5±1.0 | 2.6±1.3 | 88% (10/10 runs) | 1.1±0.7 | 21564.3±9294.8 | 6835.3±978.9 | 0.0134±0.0047 | 1.00±0.00 |
| support-triage | compose | 10 | 16.7±8.9 | 6.6±5.5 | 6.7±7.2 | 28% (10/10 runs) | 1.0 | 30511.1±37549.1 | 4970.7±1628.9 | 0.0158±0.0108 | 1.00±0.00 |
| support-triage | runlet | 10 | 6.8±0.6 | 1.6±0.5 | 1.0 | 100% (10/10 runs) | 0.0 | 8540.5±3025.9 | 5424.6±716.0 | 0.0056±0.0019 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
