# Reimplementation of Data-to-Text Generation with Content Selection and Planning
- original authors: Ratish Puduppully, Li Dong, Mirella Lapata
- [original paper](https://arxiv.org/pdf/1809.00582.pdf)
- [original code](https://github.com/ratishsp/data2text-plan-py) uses [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
- my aim is to use [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) or to write custom `python` + `tensorflow` code from scratch
<br>
- [rotowire dataset paper](https://arxiv.org/pdf/1707.08052.pdf), [rotowire dataset data](https://github.com/harvardnlp/boxscore-data)

# Dataset
- 30 teams play in the NBA, 2 of them originate from the same city, Los Angeles Lakers and Los Angeles Clippers
- train dataset contains 3398 matches out of which 9 are between Lakers and Clippers
- the authors of the dataset decided to name teams by their city, therefore on those 9 occasions teams are named `Los Angeles` and `LA` => team names are after HOME/AWAY split normalized to `LA`
- for each match there are 15 records about the team stats for each team (30 in total)
- for each match there are 614.64 records on average about the player stats
