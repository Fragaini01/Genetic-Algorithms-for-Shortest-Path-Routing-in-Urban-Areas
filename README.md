# Genetic Algorithms for Shortest Path Routing in Urban Areas
This repository holds all the relevant files produced during the final project for the subject Advanced Methods in Applied Statisics 2025 at Københavns Universitet. 
The main objective of the project was to understand and apply genetic algorithms to find the shortest path in a town/city. In this case, the vast majority of testing was done in the Copenhagen Municipality but the code is adaptable to other places.

## Authors:
  - Miguel Alonso Mediavilla (mi.alonso.mediavilla@gmail.com)
  - Francesco Ragaini (francesco.ragaini@outlook.com)

## Running
The sequential version of the implementation can be run simply by 
$python sequential_version.py
if the initial/ending points, area (city), number of generations, population, or mutation ratio want to be changed, one can do this in the file.
This version will calculate and print the time taken, the best found path and the best theoretical path for comparison. Furthermore, it will create a HTML file that can be executed to see the path. An example of the HTML can be found in the repository. Alternatively, here is a picture of it:
![alt text](https://github.com/Fragaini01/Genetic-Algorithms-for-Shortest-Path-Routing-in-Urban-Areas/blob/main/p1pop2000g150.png)
This picture is the path found from Niels Bohr Building to Niels Bohr Institute, Copenhagen.

The parallel version was optimized for testing three different problems, whose description can be found in the file. It can be executed by:
$python parallel_version.py <problem_num> <population_size> <number_of_generations> <mutation_ratio>

