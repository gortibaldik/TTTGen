# Statistics of the RotoWire dataset

- dataset is structured in form of `.json` files
- used train, validation, test split is the same as the one provided by the authors: 3398 train, 727 validation, 728 test

### Dataset sample
- in all types of architectures these are the parts of a sample which aren't used neither for training nor for generation

| name | description |
| --- | --- |
| date | when the match was played |
| visitor name | name of the team - e.g. Lakers |
| visitor city | city of origin of the team - e.g. Los Angeles |
| home name | name of the team - e.g. Clippers |
| home city | city of origin of the team - e.g. Los Angeles |

<br>
- these are relevant as the inputs to the neural network

| name      | description |
| --------- | ----------- |
| box score | individual statistics of all the players and additional info to help assign the particular player to team |
| visitor line score | visitor team statistics |
| home line score | home team statistics |

### Dataset statistics
- the gathered summary stats are before cleaning (e.g. transforming `Curry` -> `Stephen_Curry`, transforming numbers from words to digits, preprocessing some non valid words in the original dataset), after cleaning and after applying bpe on cleaned summaries
- summary statistics :

| dataset                   | max summary length | min summary length  | average summary length | number of samples |
| ------------------------- |--------------------|---------------------|------------------------| ----------------- |
| __train__ (no clean)      | 762                | 149                 | 334.406                | 3397              |
| __train__ (clean)         | 752                | 144                 | 327.137                | 3397              |
| __train__ (bpe)           | 870                | 152                 | 358.733                | 3397              |
| ------------------------- | ------------------ | ------------------- | ---------------------- | ----------------- |
| __validation__ (no clean) | 813                | 154                 | 339.972                | 727               |
| __validation__ (clean)    | 803                | 150                 | 332.613                | 727               |
| __validation__ (bpe)      | 854                | 151                 | 364.765                | 727               |
| ------------------------- | ------------------ | ------------------- | ---------------------- | ----------------- |
| __test__ (no clean)       | 782                | 149                 | 346.832                | 728               |
| __test__ (clean)          | 773                | 145                 | 339.368                | 728               |
| __test__ (bpe             | 814                | 148                 | 373.155                | 728               |
<br>
- right now no preprocessing of tables takes place
- table statistics :

| dataset        | max table length | min table length  | average table length | number of samples |
| -------------- |------------------|-------------------|----------------------| ----------------- |
| __train__      | 750              | 558               | 644.646              | 3397              |
| __validation__ | 702              | 582               | 644.657              | 727               |
| __test__       | 702              | 558               | 645.033              | 728               |
<br>

#### Table cell types
- only used table parts are counted
- 24 cell types for individual statistics (the line score part) e.g. `PLAYER_TEAM`, `AST`_assists_, `BLK` _blocks_
- 15 cell types for team statistics (the box score part) e.g. `TEAM_NAME`, `TEAM_REB` _total team rebounds_

#### Player statistics
- number of unique players mentioned in box scores:

| occurrences | train dataset | validation dataset | test dataset |
| ----------- | ------------- | ------------------ | ------------ |
| >=1         | 668           | 650                | 660          |
| >=5         | 651           | 596                | 595          |
<br>
- number of unique players mentioned in summaries:

| occurrences | train dataset | validation dataset | test dataset |
| ----------- | ------------- | ------------------ | ------------ |
| >=1         | 552           | 461                | 474          |
| >=5         | 456           | 299                | 317          |
<br>
- before looking at the data 2 assumptions about them were made:

  1. there would be significantly more players in the match stats than in summaries <br>
    - the assumption wasn't correct, the wast majority (82.6%, 70.9% and 71.8% respectively) of the players mentioned in the stats are mentioned in some summary of the respective dataset

  2. star players would be represented more in the summaries <br>
    - the assumption was correct, Russell Westbrook was the most mentioned player in both training and validation summaries (1., 2., 1.)
    - the extraction of the player names from summaries didn't mind anaphoras, references. E.g. if Stephen Curry was to be represented in the summary either "Stephen Curry" or "Curry" or "Stephen" had to be present in the summary
<br>
- the most mentioned players in the summaries:

| Position | Train             | Validation            | Test              |
|----------|-------------------|-----------------------|-------------------|
| 1.       | Russell Westbrook | LeBron James          | Russell Westbrook |
| 2.       | Stephen Curry     | Russell Westbrook     | DeMarcus Cousins  |
| 3.       | LeBron James      | Stephen Curry         | James Harden      |
| 4.       | Anthony Davis     | DeMarcus Cousins      | Stephen Curry     |
| 5.       | James Harden      | Kyrie Irving          | Kevin Durant      |
| 6.       | DeMarcus Cousins  | Kyle Lowry            | Anthony Davis     |
| 7.       | Damian Lillard    | James Harden          | John Wall         |
| 8.       | Kevin Durant      | John Wall             | Chris Paul        |
| 9.       | John Wall         | Carmelo Anthony       | Isaiah Thomas     |
| 10.      | DeMar DeRozan     | DeMar DeRozan         | Dwyane Wade       |
| 11.      | Kyrie Irving      | Anthony Davis         | Damian Lillard    |
| 12.      | Kyle Lowry        | Kevin Durant          | Carmelo Anthony   |
| 13.      | Isaiah Thomas     | Chris Paul            | LeBron James      |
| 14.      | Chris Paul        | Kemba Walker          | Kyrie Irving      |
| 15.      | Jimmy Butler      | Kevin Love            | Bradley Beal      |
| 16.      | Kemba Walker      | Giannis Antetokounmpo | Kawhi Leonard     |
| 17.      | Klay Thompson     | Kawhi Leonard         | Jimmy Butler      |
| 18.      | Dwyane Wade       | Blake Griffin         | Blake Griffin     |
| 19.      | Andre Drummond    | Brook Lopez           | DeMar DeRozan     |
| 20.      | Kawhi Leonard     | Andre Drummond        | Klay Thompson     |

##### Transformations
- assumption : it would be easier for the neural network to learn how to use player names in the summaries if it wouldn't have to distinguish between "Luc Mbah a Moute", "Moute", "Mbah a Moute" etc. but just learn to use token `Luc_Mbah_a_Moute` (or more commonly merge "Stephen", "Curry", "Stephen Curry" to `Stephen_Curry`)
- data would become denser
- references are resolved, if only player's surname is used it's transformed into token special for the player

```txt
Jusuf Nurkic -> Jusuf_Nurkic
McCollum -> CJ_McCollum
Nurkic -> Jusuf_Nurkic
Nikola Jokic -> Nikola_Jokic
```

- the tool for resolving references looks only to players on the team line-up for the game, it enables reasonable solutions for sentences like (LeBron hasn't played in this game):

```txt
Gordon Hayward put up a LeBron James-esque line of 27 points , 7 rebounds , and 5 assists in 36 minutes .

Gordon_Hayward put up a LeBron James-esque line of 27 points , 7 rebounds , and 5 assists in 36 minutes .
```

##### Problems in player name domain
- there were numerous problems with data correctness, possibly many aren't discovered yet ("Stephen Curry" is often referred to as "Steph", "Dwyane Wade" multiple times spelled as "Dwayne Wade", "Jonathon Simmons" as "Jonathan Simmons", "C.J. Collum" as "C.J . Collum")
- many times names of coaches and stuff is mentionned although it isn't present in the table data (as well as names of legendary figures like Michael Jordan)
- many player have multiple names like "Luc Mbah a Moute", who is sometimes "Luc Richard Mbah a Moute"

#### Transformations of city names
- there are only 29 names of cities and 30 names of teams
- in the city domain, only Los Angeles, Golden State, New York, San Antonio and New Orleans are multi-token names
- in the teams domain, only City Thunder and Trail Blazers are multi-token names
- therefore the increased density of the names doesn't pay off the effort for doing and testing the work
- therefore no transformation of the city names is done

#### Token statistics
- these are tokens extracted from cleaned summaries
- assumption: a neural network learns usages of tokens which are present more than or equal to 5 times in the training dataset 

| dataset    | Unique tokens | Tokens with >= 5 occurrences absolute | Tokens with >= 5 occurrences relative |
| ---------- | ------------- | --------------------------------------| ------------------------------------- |
| train      | 9771          | 4153                                  | 42.503%                               |
| validation | 5620          | 2312                                  | 41.139%                               |
| test       | 5735          | 2356                                  | 41.081%                               |

- another interesting statistics is how many tokens from the validation and test dataset can be learnt in the training dataset

| dataset    | Overlap with train | >= 5 occurrences from train overlap |
| ---------- | ------------------ | ----------------------------------- |
| validation | 88.132%            | 66.601%                             |
| test       | 87.428%            | 65.684%                             |

- therefore a decision was made to apply Byte Pair Encoding on summaries to increase the density of the data
- BPE is learned only from the train dataset
- __number of merges__ : defines how many times the BPE iteration takes place
- __BPE iteration__: during one iteration, the most frequent pair of tokens is found and replaced by a special token representing the combination of the tokens
- 3 number of merges were tried:

| dataset    | number of merges | Unique tokens | Tokens with >= 5 occurrences absolute | Tokens with >= 5 occurrences relative | 
| ---------- | ---------------- | ------------- | ------------------------------------- | ------------------------------------- |
| train      | 1500             | 1621          | 1527                                  | 94.20%                                |
| train      | 2000             | 2097          | 1964                                  | 93.66%                                |
| train      | 2500             | 2555          | 2377                                  | 93.03%                                |
| ---------- | ---------------- | ------------- | ------------------------------------- | ------------------------------------- |
| validation | 1500             | 1552          | 1412                                  | 90.98%                                |
| validation | 2000             | 2001          | 1786                                  | 89.26%                                |
| validation | 2500             | 2422          | 2101                                  | 86.75%                                |
| ---------- | ---------------- | ------------- | ------------------------------------- | ------------------------------------- |
| test       | 1500             | 1545          | 1405                                  | 90.94%                                |
| test       | 2000             | 1995          | 1787                                  | 89.57%                                |
| test       | 2500             | 2411          | 2108                                  | 87.43%                                |
| ---------- | ---------------- | ------------- | ------------------------------------- | ------------------------------------- |
| overall    | 1500             | 1641          | 1549                                  | 94.39%                                |
| overall    | 2000             | 2113          | 1999                                  | 94.61%                                |
| overall    | 2500             | 2575          | 2421                                  | 94.02%                                |

- based on the presented stats, the decision is to use Byte Pair Encoding with 2000 merges on all the tokens except the preprocessed names of players, which will be left as is after preprocessing
