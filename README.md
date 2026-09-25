# SPRINT

SPRINT: Single-step Generative Recommendation via Average Probability Velocity


```

One command per dataset:

| dataset                      | command                                   |
|------------------------------|-------------------------------------------|
| Amazon 2014 Beauty           | `python main.py --experiment=beauty`      |
| Amazon 2014 Sports           | `python main.py --experiment=sports`      |
| Amazon 2014 Toys             | `python main.py --experiment=toys`        |
| Amazon 2023 Scientific       | `python main.py --experiment=scientific`  |
| Amazon 2023 Instruments      | `python main.py --experiment=instrument`  |
| Amazon 2023 Games            | `python main.py --experiment=games`       |
| Amazon 2023 Arts             | `python main.py --experiment=arts`        |
| Yelp                         | `python main.py --experiment=yelp`        |


Run in the background and keep the console output:

```bash
mkdir -p nohup
nohup python main.py --experiment=beauty > nohup/beauty.out 2>&1 &
```

