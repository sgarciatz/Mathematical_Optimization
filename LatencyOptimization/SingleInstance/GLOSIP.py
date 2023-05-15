import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from mip import *
import pandas as pd



class GLOSIP(object): 

    def __init__(self, configuration):
        self.T      = configuration['T']
        self.D      = configuration['D']
        self.M      = configuration['M']
        self.H      = configuration['H']
        self.GAMMA  = configuration['GAMMA']
        self.model  = Model(solver_name='CBC')


    def showDeploymentMatrix(self, background_img: str = None):
        cmap = ListedColormap(['#444444', '#FFFFFF'], name='from_list', N=None)
        try:
            plt.imshow(plt.imread(background_img), extent=[0, self.T[1], self.T[0], 0])
        except:
            pass
        sns.heatmap(np.array(self.D).astype(np.int8), cmap=cmap, linecolor='gainsboro', linewidths=.1, alpha=0.5)

        plt.show()

    def showHeatMaps(self, background_img: str = None) :
        rows = math.ceil(len(self.H[:])/2)
        fig, ax = plt.subplots(rows, 2, figsize=(15, 15))
        for heatmap, i in zip(self.H[:], range(len(self.H[:]))):
            print(heatmap)
            ax[i//2][i%2].title.set_text(self.M[i])
            sns.heatmap(heatmap[:], cmap="hot", vmin = 0, vmax= np.amax(self.H), ax=ax[i//2][i%2], linecolor='gainsboro', linewidths=.1, alpha=0.5)
            try:
                ax[i//2][i%2].imshow(plt.imread(background_img), extent=[0, self.T[1], self.T[0], 0])
            except:
                pass
        if (len(self.H[:]) % 2 == 1): fig.delaxes(ax[-1][-1])
        plt.show()


    def initializeModel(self):
        #Set the decision variables
        for microservice in self.M:
            for i in range(self.T[0]):
                for j in range(self.T[1]):
                    self.model.add_var(f"s_{microservice},d{i},{j}", var_type=BINARY)

        decision_variables = np.array(self.model.vars)

        decision_variables_aux = decision_variables.reshape((len(self.M), -1)).T

        for i in range(decision_variables_aux.shape[0]):   
            constraint = xsum(decision_variables_aux[i]) <= self.D[i//self.T[1]][i%self.T[1]]
            self.model.add_constr(constraint, name=f'Capacity of d_{i//len(self.M)}{i%len(self.M)} not surpased')

        decision_variables_aux = decision_variables.reshape((len(self.M),-1))
        for i in range(len(self.M)):
            # constraint = xsum(decision_variables_aux[i]) >= 1
            constraint = xsum(decision_variables_aux[i]) == 1
            self.model.add_constr(constraint, name=f'Microservice {self.M[i]} deployed')

        decision_variables_aux = decision_variables.reshape((len(self.M), self.T[0], self.T[1]))
        objective_function = 0
        objective_function = \
        [ \
            [ self.D[i][j] * \
                [ \
                    [  self.D[ii][jj] * \
                        [ decision_variables_aux[k][ii][jj] * self.H[k][i][j] * self.GAMMA[i*self.T[1]+j][ii*self.T[1]+jj] \
                        for k in range(len(self.M))] \
                    for ii in range(self.T[0])] \
                for jj in range(self.T[1])] \
            for j in range(self.T[1])] \
        for i in range (self.T[0])]

        objective_function = xsum(self.flatten(objective_function))
        self.model.objective = minimize(objective_function)

    def solve(self):
        self.model.optimize()
        return pd.DataFrame([{"Decision Variable": var.name, "Value": var.x} for var in self.model.vars])

    def printSolution(self, background_img: str = None):
        solution_df = pd.DataFrame([{"Decision Variable": var.name, "Value": var.x} for var in self.model.vars])
        solution_ndarray = solution_df.to_numpy()
        solution_ndarray = solution_ndarray.reshape((len(self.M), self.T[0]*self.T[1], -1))

        deployment_matrix = solution_ndarray[:,:,1]
        #print(deployment_matrix.reshape(np.shape(H)))

        deployment_matrix = deployment_matrix.reshape((len(self.M), self.T[0], self.T[1]))
        deployment_matrix = [m *(i+1) for m, i in zip(deployment_matrix[:], range(deployment_matrix.shape[0]))]
        deployment_matrix = np.add.reduce(deployment_matrix[:] )

        colors = ['#FFFFFF', '#FF2D01','#0184FF', '#FFBA01','#B601FF', '#FFF701', '#9BFF01', '#01FFDC','#010DFF', '#FF01E0']
        cmap = ListedColormap(colors[0:len(self.M)+1], name='from_list', N=None)

        plt.figure(figsize = (24,8))
        ax = sns.heatmap(deployment_matrix.astype(np.int8), cmap=cmap, linecolor='gainsboro', linewidths=.1, alpha=0.6)
        plt.title('Deployment Matrix')

        colorbar = ax.collections[0].colorbar
        n = len(self.M) + 1


        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(len(self.M) + 1)])
        colorbar.set_ticklabels(['Empty'] + self.M)
        try:
            plt.imshow(plt.imread(background_img), extent=[0, self.T[1], self.T[0], 0])
        except:
            pass
        plt.show()


    def solutionLongFormat(self, scenario_name: str):
            proposed_solution = np.array([decision_variable.x for decision_variable in self.model.vars])

            proposed_solution = proposed_solution.reshape((len(self.M), self.T[0], self.T[1]))

            proposed_solution = \
            [[[self.D[i][j] * [[self.D[ii][jj] * proposed_solution[k][ii][jj] * (self.H[k][i][j] if self.H[k][i][j] <= 1 else 1)  * self.GAMMA[i*self.T[1]+j][ii*self.T[1]+jj]
                            for jj in range(self.T[1])]
                        for ii in range (self.T[0])] 
                    for j in range(self.T[1])]
                for i in range (self.T[0])]
            for k in range(len(self.M))]

            proposed_solution = [self.flatten(sublist) for sublist in proposed_solution]

            for index, sublist in enumerate(proposed_solution):
                proposed_solution[index] = [element for element in sublist if element != 0]
                
            dataset = []
            for ms, sublist in zip(self.M, proposed_solution):
                    for cost in sublist:
                        dataset.append([scenario_name, ms, cost])

            return dataset
    
    def resetModel(self):
        self.model  = Model(solver_name='CBC')


    def flatten(self, xs):
        result = []
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(self.flatten(x))
        else:
            result.append(xs)
        return result
    


from tkinter import Tk
from tkinter import filedialog
import json

configuration = json.load(open(r'./test_005.json'))

my_model = GLOSIP(configuration=configuration)
my_model.showDeploymentMatrix('scenario3.png')
my_model.showHeatMaps('scenario3.png')
my_model.initializeModel()
my_model.solve()
my_model.printSolution('scenario3.png')

scenarios = my_model.solutionLongFormat('Test_05')



configuration = json.load(open(r'./test_006.json'))

my_model = GLOSIP(configuration=configuration)
my_model.showHeatMaps('scenario3.png')
my_model.initializeModel()
my_model.solve()
my_model.printSolution('scenario3.png')

scenarios += my_model.solutionLongFormat('Test_06')

configuration = json.load(open(r'./test_007.json'))

my_model = GLOSIP(configuration=configuration)
my_model.showHeatMaps('scenario3.png')
my_model.initializeModel()
my_model.solve()
my_model.printSolution('scenario3.png')

scenarios += my_model.solutionLongFormat('Test_07')

print(scenarios)


df = pd.DataFrame(data = scenarios,
                  columns = ['Scenario','Microservice', 'Hops'])
df['Hops'] = df['Hops'].subtract(1)
print(df)


medianprops = {'linestyle':'-', 'linewidth':2, 'color':'lightcoral'}
meanpointprops = {'marker':'D', 'markeredgecolor':'forestgreen','markerfacecolor':'forestgreen'}

sns.set_theme(style="ticks", palette="pastel")


fig, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(data= df, x="Microservice", y="Hops", hue="Scenario",
            meanprops = meanpointprops, showmeans=True, 
            flierprops={"marker": "x"},
            medianprops=medianprops)

plt.xticks(range(len(my_model.M)), my_model.M, rotation=45)
plt.yticks(range(0, int(df.loc[df['Hops'].idxmax()]['Hops'])+2))
plt.xlabel('Microservices', fontsize=16)
plt.ylabel('Latency (nº hops)', fontsize=16)
plt.title('Medium sized Scenario', pad=30, fontsize=24)
plt.legend(fontsize=14)
plt.show()
