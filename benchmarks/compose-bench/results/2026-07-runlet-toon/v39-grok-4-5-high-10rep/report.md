# compose-bench report

Model: `x-ai/grok-4.5`. Token columns sum all requests in a run; peak ctx is the largest single request (input + cached + output). Cost is provider-reported (blank when OpenRouter omitted it).

| scenario | arm | runs | wall s | model reqs | tool calls | compose share | compose fails | total tokens | peak ctx | cost $ | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| calendar-scheduling | compose | 10 | 17.1±4.1 | 4.1±0.3 | 3.1±0.3 | 65% (10/10 runs) | 1.0 | 16829.0±1242.6 | 5555.9±286.8 | 0.0177±0.0042 | 1.00±0.00 |
| calendar-scheduling | runlet | 10 | 28.1±15.2 | 4.3±1.5 | 3.4±1.4 | 74% (10/10 runs) | 1.0±0.6 | 30884.3±18621.2 | 9274.5±3046.6 | 0.0293±0.0179 | 1.00±0.00 |
| crm-hygiene | compose | 10 | 17.0±1.6 | 4.0 | 3.0 | 67% (10/10 runs) | 1.0 | 15808.6±491.5 | 5009.9±138.4 | 0.0136±0.0009 | 1.00±0.00 |
| crm-hygiene | runlet | 10 | 15.4±3.0 | 3.0±0.4 | 2.1±0.5 | 100% (10/10 runs) | 1.1±0.5 | 19272.7±4101.8 | 7335.5±753.1 | 0.0167±0.0038 | 1.00±0.00 |
| log-incident | compose | 10 | 12.8±3.4 | 4.5±0.7 | 4.0±2.0 | 55% (10/10 runs) | 1.0 | 22282.8±5057.9 | 7455.4±1050.6 | 0.0137±0.0026 | 1.00±0.00 |
| log-incident | runlet | 10 | 13.8±4.1 | 4.2±1.0 | 3.2±1.0 | 97% (10/10 runs) | 0.2±0.4 | 31616.6±8672.1 | 10357.9±2350.8 | 0.0163±0.0045 | 1.00±0.00 |
| revenue-report | compose | 10 | 7.8±0.5 | 4.0 | 3.0 | 67% (10/10 runs) | 1.0 | 12672.8±804.2 | 3690.4±190.9 | 0.0076±0.0010 | 1.00±0.00 |
| revenue-report | runlet | 10 | 10.3±1.4 | 3.8±0.6 | 2.8±0.6 | 79% (10/10 runs) | 1.0 | 23105.0±5005.2 | 6874.3±377.3 | 0.0127±0.0017 | 1.00±0.00 |
| support-triage | compose | 10 | 10.1±1.5 | 4.1±0.3 | 3.1±0.3 | 68% (10/10 runs) | 1.0 | 13248.1±1588.9 | 4038.1±179.8 | 0.0101±0.0013 | 1.00±0.00 |
| support-triage | runlet | 10 | 5.6±0.3 | 1.8±0.4 | 1.0 | 100% (10/10 runs) | 0.0 | 9227.2±2657.5 | 5461.6±721.1 | 0.0070±0.0020 | 1.00±0.00 |

## composition arms vs granular (per scenario)

| scenario | arm | Δ wall | Δ model reqs | Δ total tokens | Δ cost | Δ accuracy |
|---|---|---|---|---|---|---|
