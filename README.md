# Advent of Code

This notebook contains my solutions for the 2021 version of [Advent of Code](https://adventofcode.com/2021).

![Test Notebook](https://github.com/jnslk/AoC2021/workflows/test%20notebook/badge.svg)


## Imports and Dataimport


```python
from collections import Counter, deque, defaultdict
from statistics import median, mean
from math import floor, ceil
from typing import Tuple, Set, cast, List
import re

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




    17884



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
    for pair in points:
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        if x1 == x2 or y1 == y2:
            for x in range(min(x1,x2),max(x1,x2)+1):
                for y in range(min(y1,y2),max(y1,y2)+1):
                    overlapping[(x,y)] += 1
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
test6_1_input = '3,4,3,1,2'

test6_1_output1 = 26
test6_1_output2 = 5934

def simulate_lanternfish(fish, n=80) -> int:
    fish_population = Counter(fish)
    for day in range(n):
        fish_population = simulate_day(fish_population)
    return sum(fish_population.values())

def simulate_day(fish):
    new_population = Counter()
    for f in range(1, 9):
        new_population[f-1] = fish[f]
    new_population[6] += fish[0]
    new_population[8] = fish[0]
    return new_population

assert simulate_lanternfish([*map(int, test6_1_input.split(','))], 18) == test6_1_output1
assert simulate_lanternfish([*map(int, test6_1_input.split(','))]) == test6_1_output2


input6 = data(6,int,sep=',')

simulate_lanternfish(input6)
```




    373378



## Part 2


```python
test6_2_output = 26984457539

assert simulate_lanternfish([*map(int, test6_1_input.split(','))], 256) == test6_2_output

simulate_lanternfish(input6, 256)
```




    1682576647495



# Day 7: The Treachery of Whales

## Part 1


```python
test7_1_input = '16,1,2,0,4,2,7,1,2,14'

test7_1_output = 37

def minimum_fuel_cost(crabs) -> int:
    cost = 0
    target = median(crabs)
    for crab in crabs:
        cost += abs(crab - target)
    return int(cost)

assert minimum_fuel_cost([*map(int ,test7_1_input.split(','))]) == test7_1_output

input7 = data(7,int,sep=',')

minimum_fuel_cost(input7)
```




    347509



## Part 2


```python
test7_2_output = 168

def minimum_fuel_cost2(crabs) -> int:
    cost_floor = 0
    cost_ceil = 0
    target_mean_floor = int(floor(mean(crabs)))
    target_mean_ceil = int(ceil(mean(crabs)))
    
    for crab in crabs:
        cost_floor += (abs(crab - target_mean_floor)) * (abs(crab - target_mean_floor)+1) / 2
        cost_ceil += (abs(crab - target_mean_ceil)) * (abs(crab - target_mean_ceil)+1) / 2
    return int(min(cost_floor, cost_ceil))

assert minimum_fuel_cost2([*map(int ,test7_1_input.split(','))]) == test7_2_output

input7 = data(7,int,sep=',')

minimum_fuel_cost2(input7)
```




    98257206



# Day 8: Seven Segment Search

## Part 1


```python
test8_1_input = '''be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce'''

test8_1_output = 26

#def parse_seven_segment(line):
#    line.replace('| ', '')
#    return line.split()

def parse_seven_segment(line):
    line.strip()
    return line.split('|')

def count_unique_digits(segments) -> int:
    count = 0
    for segment in segments:
        for digit in segment[1].split():
            if len(digit) <= 4 or len(digit) == 7:
                count += 1
    return count

assert count_unique_digits([*map(parse_seven_segment, test8_1_input.split('\n'))]) == test8_1_output

input8 = data(8, parse_seven_segment)

count_unique_digits(input8)
```




    355



## Part 2


```python
test8_2_output= 61229

def occurence_counter(s):
    return Counter(list(s.replace(" ", "")))

def occurence_pattern(s, ctr):
    return tuple(sorted([ctr[x] for x in s]))

canonical_pattern = "abcefg cf acdeg acdfg bdcf abdfg abdefg acf abcdefg abcdfg"
canonical_dict = occurence_counter(canonical_pattern)

translator = {}
for i, x in enumerate(canonical_pattern.split(" ")):
    translator[occurence_pattern(x, canonical_dict)] = i

def process_line(ls):
    outputs = ls[1].strip()
    occ_dict = occurence_counter(ls[0])
    return [translator[occurence_pattern(x, occ_dict)] for x in outputs.split(" ")]

def count_all_digits(segments) -> int:
    count = 0
    for segment in segments:
        p = process_line(segment)
        count += int("".join([str(x) for x in p]))
    return count

assert count_all_digits([*map(parse_seven_segment, test8_1_input.split('\n'))]) == test8_2_output

input8 = data(8, parse_seven_segment)

count_all_digits(input8)
```




    983030



# Day 9: Smoke Basin

## Part 1


```python
test9_1_input = '''2199943210
3987894921
9856789892
8767896789
9899965678'''

test9_1_output = 15

def find_lowpoints(heightmap) -> int:
    Grid = []
    for line in heightmap:
        Grid.append([int(x) for x in line])
    rows = len(Grid)
    cols = len(Grid[0])
    row_deltas = [-1,0,1,0]
    col_deltas = [0,1,0,-1]
    riskscore = 0
    for r in range(rows):
        for c in range(cols):
            lowpoint = True
            for delta in range(len(row_deltas)):
                rr = r + row_deltas[delta]
                cc = c + col_deltas[delta]
                if 0<=rr<rows and 0<=cc<cols and Grid[rr][cc]<=Grid[r][c]:
                    lowpoint = False
            if lowpoint:
                riskscore += Grid[r][c]+1
    return riskscore

assert find_lowpoints(test9_1_input.split('\n')) == test9_1_output

input9 = data(9)

find_lowpoints(input9)
```




    425



## Part 2


```python
test9_2_output = 1134

def find_basins(heightmap) -> int:
    Grid = []
    for line in heightmap:
        Grid.append([int(x) for x in line])
    rows = len(Grid)
    cols = len(Grid[0])
    row_deltas = [-1,0,1,0]
    col_deltas = [0,1,0,-1]
    basins = []
    seen = set()
    for r in range(rows):
        for c in range(cols):
            if (r,c) not in seen and Grid[r][c]!=9:
                size = 0
                Q = deque()
                Q.append((r,c))
                while Q:
                    (r, c) = Q.popleft()
                    if (r, c) in seen:
                        continue
                    seen.add((r,c))
                    size += 1
                    for delta in range(len(row_deltas)):
                        rr = r+row_deltas[delta]
                        cc = c+col_deltas[delta]
                        if 0<=rr<rows and 0<=cc<cols and Grid[rr][cc]!=9:
                            Q.append((rr,cc))
                basins.append(size)
    basins.sort()                
    return basins[-1] * basins[-2] * basins[-3]

assert find_basins(test9_1_input.split('\n')) == test9_2_output

input9 = data(9)

find_basins(input9)
```




    1135260



# Day 10: Syntax Scoring

## Part 1


```python
test10_1_input = '''[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]'''

test10_1_output = 26397

match = {')': '(', ']': '[', '}': '{', '>': '<'}

penalty = {')': 3, ']': 57, '}': 1197, '>': 25137}

def syntax_error_score(navigation_lines) -> int:
    error_score = 0
    for line in navigation_lines:
        stack = []
        for symbol in line:
            if symbol in set('([<{'):
                stack.append(symbol)
            if symbol in set('}])>'):
                previous = stack.pop()
                if match[symbol] == previous:
                    continue
                else:
                    error_score += penalty[symbol]
    return error_score

assert syntax_error_score(test10_1_input.split('\n')) == test10_1_output

input10 = data(10)

syntax_error_score(input10)
```




    462693



## Part 2


```python
test10_2_output = 288957

score = {'(': 1, '[': 2, '{': 3, '<':4}

def auto_complete_score(navigation_lines) -> int:
    completion_scores = []
    for line in navigation_lines:
        stack = []
        for symbol in line:
            if symbol in set('([<{'):
                stack.append(symbol)
            elif not stack or stack.pop() != match[symbol]:
                stack = None
                break
                
        if stack:
            subtotal = 0

            for symbol in stack[::-1]:
                subtotal = 5 * subtotal + score[symbol]

            completion_scores.append(subtotal)
                    
    
    return median(completion_scores)

assert auto_complete_score(test10_1_input.split('\n')) == test10_2_output

input10 = data(10)

auto_complete_score(input10)
```




    3094671161



# Day 11: Dumbo Octopus

## Part 1


```python
test11_1_input = '''5483143223
2745854711
5264556173
6141336146
6357385478
4167524645
2176841721
6882881134
4846848554
5283751526'''

test11_1_output = 1656

rows = 10
cols = 10

def flash(r, c):
    global count
    global Grid
    count += 1
    Grid[r][c] = -1
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            rr = r+dr
            cc = c+dc
            if 0<=rr<rows and 0<=cc<cols and Grid[rr][cc]!=-1:
                Grid[rr][cc] += 1
                if Grid[rr][cc] >= 10:
                    flash(rr,cc)



def count_flashes(energy_levels, n=100) -> int:
    global count
    global Grid
    Grid = []
    for line in energy_levels:
        Grid.append([int(x) for x in line])
    rows = len(Grid)
    cols = len(Grid[0])
    count = 0

    for step in range(n):
        for r in range(rows):
            for c in range(cols):
                Grid[r][c] += 1
        for r in range(rows):
            for c in range(cols):
                if Grid[r][c] == 10:
                    flash(r,c)
        for r in range(rows):
            for c in range(cols):
                if Grid[r][c] == -1:
                    Grid[r][c] = 0
                
    return(count)

assert count_flashes(test11_1_input.split()) == test11_1_output
    
input11 = data(11)

count_flashes(input11)
```




    1599



## Part 2


```python
test11_2_output = 195

rows = 10
cols = 10

def flash(r, c):
    global count
    global Grid
    count += 1
    Grid[r][c] = -1
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            rr = r+dr
            cc = c+dc
            if 0<=rr<rows and 0<=cc<cols and Grid[rr][cc]!=-1:
                Grid[rr][cc] += 1
                if Grid[rr][cc] >= 10:
                    flash(rr,cc)

def synchronized_flashes(energy_levels) -> int:
    global Grid
    Grid = []
    for line in energy_levels:
        Grid.append([int(x) for x in line])
    step = 0
    while True:
        step += 1
        for r in range(rows):
            for c in range(cols):
                Grid[r][c] += 1
        for r in range(rows):
            for c in range(cols):
                if Grid[r][c] == 10:
                    flash(r, c)
        done = True
        for r in range(rows):
            for c in range(cols):
                if Grid[r][c] == -1:
                    Grid[r][c] = 0
                else:
                    done = False
        if done:
            return step
        
assert synchronized_flashes(test11_1_input.split()) == test11_2_output

input11 = data(11)

synchronized_flashes(input11)
```




    418



# Day 12: Passage Pathing

## Part 1


```python
test12_1_input1 = '''start-A
start-b
A-c
A-b
b-d
A-end
b-end'''

test12_1_input2 = '''dc-end
HN-start
start-kj
dc-start
dc-HN
LN-dc
HN-end
kj-sa
kj-HN
kj-dc'''

test12_1_input3 = '''fs-end
he-DX
fs-he
start-DX
pj-DX
end-zg
zg-sl
zg-pj
pj-he
RW-he
fs-DX
pj-RW
zg-RW
start-pj
he-WI
zg-he
pj-fs
start-RW'''

test12_1_output1 = 10
test12_1_output2 = 19
test12_1_output3 = 226

def parse_edges(line):
    return line.split('-')

def count_paths(edges, seen=[], cave='start') -> int:
    global neighbours
    neighbours = defaultdict(list)
    for edge in edges:
        neighbours[edge[0]] += [edge[1]]
        neighbours[edge[1]] += [edge[0]]
    return count(seen, cave)

def count(seen=[], cave='start'):
    if cave == 'end': return 1
    if cave in seen:
        if cave == 'start': return 0
        if cave.islower():
            return 0
    return sum(count(seen+[cave], n) for n in neighbours[cave])

assert count_paths([*map(parse_edges, test12_1_input1.split())]) == test12_1_output1
assert count_paths([*map(parse_edges, test12_1_input2.split())]) == test12_1_output2
assert count_paths([*map(parse_edges, test12_1_input3.split())]) == test12_1_output3

input12 = data(12, parse_edges)

count_paths(input12)
```




    5212



## Part 2


```python
test12_2_output1 = 36
test12_2_output2 = 103
test12_2_output3 = 3509

def parse_edges(line):
    return line.split('-')

def count_paths2(edges, seen=[], cave='start') -> int:
    global neighbours
    neighbours = defaultdict(list)
    for edge in edges:
        neighbours[edge[0]] += [edge[1]]
        neighbours[edge[1]] += [edge[0]]
    return count(part=2)

def count(part, seen=[], cave='start'):
    if cave == 'end': return 1
    if cave in seen:
        if cave == 'start': return 0
        if cave.islower():
            if part == 1: return 0
            else: part = 1
    return sum(count(part, seen+[cave], n)
                for n in neighbours[cave])

assert count_paths2([*map(parse_edges, test12_1_input1.split())]) == test12_2_output1
assert count_paths2([*map(parse_edges, test12_1_input2.split())]) == test12_2_output2
assert count_paths2([*map(parse_edges, test12_1_input3.split())]) == test12_2_output3

input12 = data(12, parse_edges)

count_paths2(input12)
```




    134862



# Day 13: Transparent Origami

## Part 1


```python
test13_1_input = '''6,10
0,14
9,10
0,3
10,4
4,11
6,0
6,12
4,1
0,13
10,12
3,4
3,0
8,4
1,10
2,14
8,10
9,0

fold along y=7
fold along x=5'''

test13_1_output = 17

def count_points_after_fold(points, folds) -> int:
    overlapping = 0
    target = folds[0]
    if target[0] == 'x':
        for p in points:
            if int(p[0]) > int(target[1:]):
                distance = abs(int(target[1:]) - int(p[0]))
                if [str((int(p[0]) - 2 * distance)), p[1]] in points:
                    overlapping += 1
    else:
        for p in points:
            if int(p[1]) > int(target[1:]):
                distance = abs(int(target[1:]) - int(p[1]))
                if [p[0], str((int(p[1]) - 2 * distance))] in points:
                    overlapping += 1
    return len(points) - overlapping

test_points = [x.split(',') for x in test13_1_input.split('\n\n')[0].split('\n')]
test_folds = [y[0][-1] + y[1] for y in [x.split('=') for x in test13_1_input.split('\n\n')[1].split('\n')]]

assert count_points_after_fold(test_points, test_folds) == test13_1_output

input13 = data(13, sep='\n\n')
points = [x.split(',') for x in input13[0].split('\n')]
folds = [y[0][-1] + y[1] for y in [x.split('=') for x in input13[1].split('\n')]]

count_points_after_fold(points, folds)
```




    755



## Part 2


```python
Point = Tuple[int, int]
Grid = Set[Point]

def parse_folds(folds: List[str]) -> List[Tuple[bool, int]]:
    result: List[Tuple[bool, int]] = []
    for fold in folds:
        fold_desc = re.search(r"(x|y)(\d+)", fold)
        assert fold_desc
        result.append((fold_desc.group(1) == "y", int(fold_desc.group(2))))

    return result

def print_grid_after_folds(points, folds):
    dots: Grid = {
        cast(Point, tuple(map(int, point))) for point in points
    }
    folds = parse_folds(folds)
    for fold_ins in folds:
        dots = fold_grid(dots, *fold_ins)
    print()
    print_grid(dots)
    return

def print_grid(dots: Grid):
    max_x = max(x[0] for x in dots)
    max_y = max(y[1] for y in dots)

    for y in range(max_y + 1):
        for x in range(max_x + 1):
            print("#" if (x, y) in dots else ".", end="")
        print() # newline!
    print() # space after the print
    
def fold_grid(dots: Grid, horiz: bool, val: int) -> Grid:
    result: Grid = set()
    modified_index, same_index = (1, 0) if horiz else (0, 1)

    for p in dots:
        # if being folded onto, no change
        if p[modified_index] < val:
            result.add(p)
            continue

        updated_point = [-1, -1]
        # one half of the points is unmodified
        updated_point[same_index] = p[same_index]

        # the other half changes based on its distance to the line
        updated_point[modified_index] = 2 * val - p[modified_index]

        result.add(cast(Point, tuple(updated_point)))

    return result

print_grid_after_folds(points, folds)
```

    
    ###..#....#..#...##.###..###...##...##.
    #..#.#....#.#.....#.#..#.#..#.#..#.#..#
    ###..#....##......#.#..#.###..#..#.#...
    #..#.#....#.#.....#.###..#..#.####.#.##
    #..#.#....#.#..#..#.#.#..#..#.#..#.#..#
    ###..####.#..#..##..#..#.###..#..#..###
    


# Day 14: Extended Polymerization

## Part 1


```python
test14_1_input = '''NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C'''

test14_1_output = 1588

def pairs_in_string(s):
    return [''.join(pair) for pair in zip(s[:-1], s[1:])]

def polymerization(template, insertion_rules, n=10) -> int:
    polymer = template
    rules = dict(insertion_rules)
    for step in range(n):
        pairs = pairs_in_string(polymer)
        new_polymer = ''
        for pair in pairs[:-1]:
            new_polymer = new_polymer + pair[0] + rules[pair]
        new_polymer = new_polymer + pairs[-1][0] + rules[pairs[-1]] + pairs[-1][1]
        polymer = new_polymer
    c = Counter(polymer)
    return max(c.values()) - min(c.values())

test_template = test14_1_input.split('\n\n')[0]
test_insertion_rules = [x.split(' -> ') for x in test14_1_input.split('\n\n')[1].split('\n')]

assert polymerization(test_template, test_insertion_rules) == test14_1_output

input14 = data(14, sep='\n\n')
template = input14[0]
insertion_rules = [x.split(' -> ') for x in input14[1].split('\n')]

polymerization(template, insertion_rules)
```




    2745



## Part 2


```python
test14_2_output = 2188189693529

def pairs_in_string(s):
    return [''.join(pair) for pair in zip(s[:-1], s[1:])]

def lanternfish_polymerization(template, insertion_rules, n=40) -> int:
    rules = dict(insertion_rules)
    pairs = Counter(pairs_in_string(template))
    chars = Counter(template)
    
    for step in range(n):
        for (a,b), c in pairs.copy().items():
            x = rules[a+b]
            pairs[a+b] -= c
            pairs[a+x] += c
            pairs[x+b] += c
            chars[x] += c

    return (max(chars.values()) - min(chars.values()))

assert lanternfish_polymerization(test_template, test_insertion_rules) == test14_2_output

lanternfish_polymerization(template, insertion_rules)
```




    3420801168962



# Day 15: Chiton

## Part 1


```python

```
