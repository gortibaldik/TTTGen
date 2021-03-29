# Statistics of the RotoWire dataset

- dataset is structured in form of `.json` files
- used train, validation, test split is the same as the one provided by the authors: 3398 train, 727 validation, 728 test

### Dataset sample
- in all types of architectures these are the parts of a sample which aren't used neither for training nor for generation
<br>
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
- summary statistics :
<br>
| dataset        | max summary length | min summary length  | average summary length | number of samples |
| -------------- |--------------------|---------------------|------------------------| ----------------- |
| __train__      | 762                | 149                 | 334.406                | 3397              |
| __validation__ | 813                | 154                 | 339.972                | 727               |
| __test__       | 782                | 149                 | 346.832                | 728               |
<br>
- table statistics :
<br>
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
<br>
| train dataset | validation dataset | test dataset |
| ------------- | ------------------ | ------------ |
| 669           | 651                | 661          |
<br>
- number of unique players mentioned in summaries:
<br>
| train dataset | validation dataset | test dataset |
| ------------- | ------------------ | ------------ |
| 552           | 467                | 477          |
<br>
- before looking at the data 2 assumptions about them were made:
  1. there would be significantly more players in the match stats than in summaries
    - the assumption wasn't correct, the wast majority (82.5%, 71.7% and 72.2% respectively) of the players mentioned in the stats are mentioned in some summary of the respective dataset
  2. star players would be represented more in the summaries
    - the assumption was correct, LeBron James was the most mentioned player in both training and validation summaries, strangely in the test set he didn't make top 10, Rusell Westbrook was the highest player listed in all three top tens (2., 3., 10.)
<br>
- the most mentioned players in the summaries:
| Position | Train             | Validation            | Test              |
|----------|-------------------|-----------------------|-------------------|
| 1.       | LeBron James      | LeBron James          | Chris Paul        |
| 2.       | Russell Westbrook | Kyrie Irving          | DeMarcus Cousins  |
| 3.       | Anthony Davis     | Russell Westbrook     | James Harden      |
| 4.       | James Harden      | Kyle Lowry            | Kevin Durant      |
| 5.       | DeMarcus Cousins  | John Wall             | Anthony Davis     |
| 6.       | Damian Lillard    | Carmelo Anthony       | Dwyane Wade       |
| 7.       | DeMar DeRozan     | DeMar DeRozan         | Isaiah Thomas     |
| 8.       | John Wall         | Kevin Love            | Carmelo Anthony   |
| 9.       | Kemba Walker      | James Harden          | John Wall         |
| 10.      | Kevin Durant      | DeMarcus Cousins      | Russell Westbrook |
| 11.      | Isaiah Thomas     | Giannis Antetokounmpo | Blake Griffin     |
| 12.      | Klay Thompson     | Anthony Davis         | Kawhi Leonard     |
| 13.      | Jimmy Butler      | Derrick Rose          | Jimmy Butler      |
| 14.      | Kyrie Irving      | Chris Paul            | DeMar DeRozan     |
| 15.      | Brook Lopez       | Andre Drummond        | Andrew Wiggins    |
| 16.      | Chris Paul        | Isaiah Thomas         | Damian Lillard    |
| 17.      | Andre Drummond    | Dwight Howard         | Mike Conley       |
| 18.      | Kyle Lowry        | Brook Lopez           | Marc Gasol        |
| 19.      | Kawhi Leonard     | Jimmy Butler          | LeBron James      |
| 20.      | Marc Gasol        | Kevin Durant          | Gordon Hayward    |
