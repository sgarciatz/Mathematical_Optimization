import os
import sys
import argparse
import numpy as np
import json
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt



def calcAdjacencyMatrix(input_matrix: np.ndarray):
    rows    = input_matrix.shape[0]
    columns = input_matrix.shape[1] 
    adj_matrix = np.zeros((rows*columns, rows*columns), dtype = np.int8)
    for i in range(rows):
        for j in range(columns):
            if input_matrix[i][j] == 1:
                for ii in range(max(0,i-1), min(rows-1, i+1)+1):
                    for jj in range(max(0, j-1), min(columns-1, j+1)+1):
                        if input_matrix[ii][jj] == 1:
                            adj_matrix[i*columns+j][ii*columns+jj] = 1
    return adj_matrix

def loadJsonIntoNumpy(file_path: any):
    try:

        configuration =  json.load(open(file_path))
        T     = np.array(configuration["T"])
        D     = np.array(configuration["D"])
        M     = np.array(configuration["M"])
        H     = np.array(configuration["H"])


        return {'T'     : T, 
                'D'     : D, 
                'M'     : M, 
                'H'     : H, 
               }
    except Exception as e:
        print(e)
        print('The input file was not found or its content is not right')

def calcShortestPathCost(adjacency_matrix: np.array, original_shape: np.array):

    node_names = [ f'drone_{i//original_shape[0]},{i%original_shape[0]}' for i in range(adjacency_matrix.shape[0]) ]
    a = pd.DataFrame(adjacency_matrix, index=node_names, columns=node_names)
    g = ig.Graph.Adjacency(a)

    costs = [g.get_shortest_paths(v, output = "epath") for v in g.vs]
    costs = [[ len(column) for column in row] for row in costs]
    return np.array(costs) 
def writeToOutpuFile(file_path: any, scenario: dict):

    output_data = {item[0]: item[1].tolist() for item in scenario.items()}

    with open(file_path, 'w') as out:
        json.dump(output_data, out, indent=4)

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    scenario = loadJsonIntoNumpy(args.infile.name)

    scenario['A'] = calcAdjacencyMatrix(scenario['D'])


    scenario['GAMMA'] = calcShortestPathCost(scenario['A'], scenario['T'])


    writeToOutpuFile(args.outfile.name, scenario)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
