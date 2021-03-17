# Statistics of `WIKIBIO` dataset
- dataset contains the first sentence of an wikipedia article and the infobox (structured data). Each infobox is encoded as a list of (field name, field value) pairs
- the statistics are used for filtering all the sentences which are too long and too sparsely represented in the dataset (so they wouldn't have affected the training process even if they would have been present)

## The statistics over the tables
- the data in the table are presented as `name_1:Frantisek name_2:Trebuna ...` so the length of the data means the number of key-value pairs

```txt
Found 582659 of table
Longest table : 791
Shortest table : 1
Average table : 45.66810604487359
Percent of table bigger than 25 : 75.44292630852695
Percent of table less or equal to 25 : 24.557073691473057

Percent of table bigger than 50 : 32.95632608438212
Percent of table less or equal to 50 : 67.04367391561789

Percent of table bigger than 75 : 13.17975007680307
Percent of table less or equal to 75 : 86.82024992319693

Percent of table bigger than 100 : 5.083076035897498
Percent of table less or equal to 100 : 94.91692396410251

Percent of table bigger than 125 : 1.9155286368184479
Percent of table less or equal to 125 : 98.08447136318155
```

- based on the stats, the length 100 was selected for filtering

## The statistics over the sentences (targets)
- length means number of words

```txt
Found 582659 of summary
Longest summary : 271
Shortest summary : 1
Average summary : 26.05811289278978
Percent of summary bigger than 25 : 44.23736696764317
Percent of summary less or equal to 25 : 55.76263303235683

Percent of summary bigger than 50 : 3.151757717635873
Percent of summary less or equal to 50 : 96.84824228236413

Percent of summary bigger than 75 : 0.3195694222521235
Percent of summary less or equal to 75 : 99.68043057774787

Percent of summary bigger than 100 : 0.06127082907841465
Percent of summary less or equal to 100 : 99.93872917092159

Percent of summary bigger than 125 : 0.02059523666501333
Percent of summary less or equal to 125 : 99.97940476333498
```

- based on the stats, the length 75 was selected for filtering