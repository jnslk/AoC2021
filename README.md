# Advent of Code

This notebook contains my solutions for the 2021 version of [Advent of Code](https://adventofcode.com/2021).

![Test Notebook](https://github.com/jnslk/AoC2021/workflows/test%20notebook/badge.svg)


## Imports and Dataimport


```python
from collections import Counter

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



# Day 3: Binary Diagnostic

## Part 1


```python
test3_1_input = '''00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010'''

test3_1_output = 198

def check_power_consumption(binary_data) -> int:
    counter = [0] * len(binary_data[0])
    treshold = (len(binary_data) / 2)
    for line in binary_data:
        for i, digit in enumerate(line):
            if digit == '1':
                counter[i] += 1
    gamma = [0] * len(counter)
    for i in range(len(gamma)):
        if counter[i] > treshold:
            gamma[i] = 1
    epsilon = [0] * len(gamma)
    for i in range(len(epsilon)):
        if gamma[i] == 1:
            epsilon[i] = 0
        else:
            epsilon[i] = 1
    
    gamma = int(''.join(map(str, gamma)), 2)
    epsilon = int(''.join(map(str, epsilon)), 2)
    return gamma * epsilon

assert check_power_consumption(test3_1_input.split()) == test3_1_output

input3 = data(3)

check_power_consumption(input3)
```




    3320834



## Part 2


```python
test3_2_output = 230

def verify_life_support_rating(binary_data) -> int:
    oxygen_candidates = binary_data
    co2_candidates = binary_data
    counter = [0] * len(binary_data[0])
    for i in range(len(counter)):
        for line in oxygen_candidates:
            if line[i] == '1':
                counter[i] += 1
        if len(oxygen_candidates) > 1:
            oxygen_treshold = len(oxygen_candidates) / 2
            if counter[i] >= oxygen_treshold:
                oxygen_target = '1'
            else:
                oxygen_target = '0'
            oxygen_candidates = [x for x in oxygen_candidates if x[i] == oxygen_target]
        counter[i] = 0
        for line in co2_candidates:
            if line[i] == '1':
                counter[i] += 1
        if len(co2_candidates) > 1:
            co2_treshold = len(co2_candidates) / 2
            if counter[i] >= co2_treshold:
                co2_target = '0'
            else:
                co2_target = '1'
            co2_candidates = [x for x in co2_candidates if x[i] == co2_target]
    
    
    oxygen = int(''.join(map(str, oxygen_candidates)), 2)
    co2 = int(''.join(map(str, co2_candidates)), 2)
    
    return oxygen * co2

assert verify_life_support_rating(test3_1_input.split()) == test3_2_output

verify_life_support_rating(input3)
```




    4481199



# Day 4: Giant Squid

## Part 1


```python
test4_1_input = '''7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1

22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19

 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6

14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7'''

test4_1_output = 4512

def parse_boards(boards):
    all_boards = []
    for board in boards:
        b = []
        b.append(set([*map(int, board[:5])]))
        b.append(set([*map(int, board[5:10])]))
        b.append(set([*map(int, board[10:15])]))
        b.append(set([*map(int, board[15:20])]))
        b.append(set([*map(int, board[20:25])]))
        b.append(set([*map(int, board[::5])]))
        b.append(set([*map(int, board[1::5])]))
        b.append(set([*map(int, board[2::5])]))
        b.append(set([*map(int, board[3::5])]))
        b.append(set([*map(int, board[4::5])]))
        all_boards.append(b)
    return all_boards

def winning_board_score(numbers, boards) -> int:
    for num in numbers:
        for board in boards:
            for rowcol in board:
                rowcol.discard(num)
                if len(rowcol) == 0:
                    score = sum([*map(sum,(board[:5]))])
                    return score * num

test_nums = [*map(int,test4_1_input.split('\n\n')[0].split(','))]
test_boards = [*map(str.split,test4_1_input.split('\n\n')[1:])]
assert winning_board_score(test_nums, parse_boards(test_boards))

input4 = data(4, sep='\n\n')
nums = [*map(int, input4[0].split(','))]
boards = [*map(str.split, input4[1:])]

winning_board_score(nums, parse_boards(boards))
```




    74320



## Part 2


```python
test4_2_output = 1924

def last_board_score(numbers, boards) -> int:
    winning_boards = set()
    for num in numbers:
        for i, board in enumerate(boards):
            for rowcol in board:
                rowcol.discard(num)
                if len(rowcol) == 0:
                    winning_boards.add(i)
                    if len(winning_boards) == len(boards):
                        score = sum([*map(sum,(board[:5]))])
                        return score * num 


assert last_board_score(test_nums, parse_boards(test_boards)) == test4_2_output

input4 = data(4, sep='\n\n')
nums = [*map(int, input4[0].split(','))]
boards = [*map(str.split, input4[1:])]

last_board_score(nums, parse_boards(boards))
```

# Day 5: Hydrothermal Venture

## Part 1


```python
test5_1_input = '''0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2'''

test5_1_output = 5

def parse_lines(line):
    first, second = line.split(' -> ')
    first = first.split(',')
    second = second.split(',')
    return [*map(int,first)], [*map(int,second)]

def count_overlapping_lines(points) -> int:
    overlapping = Counter()
    lines = []
    for pair in points:
        if (pair[0][0] == pair[1][0]) or (pair[0][1] == pair[1][1]):
            lines.append(pair)
    for line in lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        for x in range(min(x1,x2),max(x1,x2)+1):
            for y in range(min(y1,y2),max(y1,y2)+1):
                overlapping[(x,y)] += 1
        
    # apply lines on grid, increment counter 
    # sweep over grid and count places with value of 2 or higher

    return sum(1 for v in overlapping.values() if v > 1)

assert count_overlapping_lines([*map(parse_lines, test5_1_input.split('\n'))]) == test5_1_output

input5 = data(5, parse_lines)

count_overlapping_lines(input5)
```




    7297



## Part 2


```python
test5_2_output = 12

def count_overlapping_lines_diagonal(points) -> int:
    overlapping = Counter()
    for pair in points:
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        dx = 1 if x2>x1 else -1
        dy = 1 if y2>y1 else -1
        if x1 == x2:
            dx = 0
        if y1 == y2:
            dy = 0
        overlapping[(x1,y1)] += 1
        while x1 != x2 or y1 != y2:
            x1 += dx
            y1 += dy
            overlapping[(x1,y1)] += 1
    return  sum(1 for v in overlapping.values() if v > 1)

assert count_overlapping_lines_diagonal([*map(parse_lines, test5_1_input.split('\n'))]) == test5_2_output

input5 = data(5, parse_lines)

count_overlapping_lines_diagonal(input5)
```




    21038



# Day 6: Lanternfish

## Part 1


```python

```
