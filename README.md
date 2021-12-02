# Advent of Code

This notebook contains my solutions for the 2021 version of [Advent of Code](https://adventofcode.com/2021).

![Test Notebook](https://github.com/jnslk/AoC2021/workflows/test%20notebook/badge.svg)


## Dataimport


```python
def data(day: int, parser=str, sep='\n') -> list:
    "Split the day's input file into sections separated by `sep`, and apply `parser` function to each."
    with open(f'../data/day{day}.txt') as f:
        sections = f.read().rstrip().split(sep)
        return list(map(parser, sections))
```

# Day 1: Sonar Sweep

## Part 1
For the first puzzle we are provided with a list of depth measurements from sonar. The task is to count the number of depth measurements that are greater than the previously measured value. 


```python
test1_1_input = '''199
200
208
210
200
207
240
269
260
263'''

test1_1_output = 7

def deeper(measurements):
    count = 0
    for i, depth in enumerate(measurements[1:]):
        if depth > measurements[i]:
            count += 1
    return count

assert deeper([*map(int, test1_1_input.split())]) == test1_1_output

input1 = data(1, int)
deeper(input1)
```




    1548



## Part 2

The second part of the challenge is to use a sliding window of 3 measurements summed together and count the number of times when the measurements in this sliding window are greater than the previous sum.


```python
test1_2_output = 5

def deeper_sliding_window(measurements):
    count = 0
    for i, depth in enumerate(measurements[3:]):
        if depth + measurements[i+1] + measurements[i+2] > sum(measurements[i:i+3]):
            count += 1          
    return count

assert deeper_sliding_window([*map(int, test1_1_input.split())]) == test1_2_output

input1 = data(1, int)
deeper_sliding_window(input1)
```




    1589



# Day 2: Dive!

## Part 1



```python
test2_1_input = '''forward 5
down 5
forward 8
up 3
down 8
forward 2'''

test2_1_output = 150

def parse_course(line) -> (str, int):
    return line.split()[0], int(line.split()[1])

def follow_course(course) -> int:
    distance = 0
    depth = 0
    for instruction, value in course:
        if instruction == 'forward':
            distance += value
        elif instruction == 'down':
            depth += value
        else:
            depth -= value
    return distance * depth

assert follow_course([*map(parse_course, test2_1_input.split('\n'))]) == test2_1_output

input2 = data(2, parse_course)

follow_course(input2)
```




    1924923



## Part 2


```python
test2_2_output = 900

def follow_complex_course(course) -> int:
    distance = 0
    depth = 0
    aim = 0
    for instruction, value in course:
        if instruction == 'forward':
            distance += value
            depth += (aim * value)
        elif instruction == 'down':
            aim += value
        else:
            aim -= value
    return distance * depth

assert follow_complex_course([*map(parse_course, test2_1_input.split('\n'))]) == test2_2_output

input2 = data(2, parse_course)

follow_complex_course(input2)
```




    1982495697



# Day 3


```python

```
