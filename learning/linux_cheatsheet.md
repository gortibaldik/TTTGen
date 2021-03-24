# Making `linux` life easier

## Using `tar`
- create `tar.gz` (compressed tar) archive : `tar -czvf <archive_name> <path to directory which should be comprimed>`
<br>
| `c`               | `z`                    | `v`     | `f`                  |
|-------------------|------------------------|---------|----------------------|
| create an archive | use `gzip` compression | verbose | specify archive name |
<br>
- extract from `tar.gz` archive: `tar -xzvf <archive_name> -C <directory where to extract>`
- extract from `tar.bz2` archive: `tar -xjvf <archive_name> -C <directory where to extract>`
<br>
| `x`                           | `C`                                       | `j`                            |
|-------------------------------|-------------------------------------------|--------------------------------|
| extract from the `gz` archive | change to directory before doing anything | extract from the `bz2` archive |

## Working with `.json` files
### Inspection of the data
- particularly in RotoWire dataset there is `.json` file with all the records on one line, meaning more than 3 000 000 records become totally unreadable
- utility `jq`, [small tutorial on it](https://www.baeldung.com/linux/jq-command-json)

### Processing of the data
- `python 3.8` `json` library by default doesn't keep the order of the elements, because it stores the collected objects to `dict` class which doesn't remember it
- therefore it is better to load the json file with `json.load(<file>, object_pairs_hook=OrderedDict)`
- `collections.OrderedDict` is dictionary that keeps the order of insertion of elements

