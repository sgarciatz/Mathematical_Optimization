import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from mip import *
import pandas as pd
import json


class GLOMIP(object):

    def __init__(self, configuration):
        self.T      = configuration['T']
        self.D      = [{'id': str(i*self.T[1]+j), 'position':[i,j]} for i, row in enumerate(configuration['D']) for j, drone in enumerate(row) if drone == 1]
        self.P      = configuration['P']
        self.M      = configuration['M']
        self.H      = configuration['H']
        self.H = np.array([[self.H[i][drone['position'][0]][drone['position'][1]] for j, drone in enumerate(self.D)] for i, microservice in enumerate(self.M)])
        self.H += 1
        self.C      = (np.array(self.P).max() * np.array(self.H).max()) + 1
        self.model = Model(sense=MINIMIZE, solver_name=GRB)
        self.model.verbose = 1
        self.model.threads = -1

    def showDeploymentMatrix(self, background_img: str = None):
        cmap = ListedColormap(['#444444', '#FFFFFF'], name='from_list', N=None)
        deployment_matrix = np.zeros((self.T[0], self.T[1]), dtype=np.int8)
        for drone in self.D:
            deployment_matrix[drone['position'][0]][drone['position'][1]] = 1
            
        sns.heatmap(deployment_matrix, cmap=cmap, linecolor='gainsboro', linewidths=.1, alpha=0.5)
        try:
            plt.imshow(plt.imread(background_img), extent=[0, self.T[1], self.T[0], 0])
        except:
            pass
        plt.show() 

    def showHeatMaps(self, background_img: str = None) :
        rows = math.ceil(len(self.H[:])/2)
        fig, ax = plt.subplots(rows, 2)
        heatmaps = np.zeros((len(self.M), self.T[0], self.T[1]), dtype=np.int8)
        for i, ms_heat in enumerate(self.H):
            for j, drone in enumerate(self.D):
                heatmaps[i][drone['position'][0]][drone['position'][1]] = ms_heat[j]
        
        for i, (heatmap, subplot) in enumerate(zip(heatmaps, np.array(ax).reshape((-1)))):
            subplot.title.set_text(self.M[i])
            sns.heatmap(heatmap, cmap="hot", vmin = 0, vmax= np.amax(self.H), ax=subplot, linecolor='gainsboro', linewidths=.1, alpha=0.5)
            try:
                ax[i//2][i%2].imshow(plt.imread(background_img), extent=[0, self.T[1], self.T[0], 0])
            except:
                pass
        if (len(self.H[:]) % 2 == 1): fig.delaxes(ax[-1][-1])
        plt.show()

    def initializeModel(self):
        # Add the decision variables
        #   x (i,j,k)       -> says if m(k) is deployed in d(i,j)
        #   z (i,j,k)       -> the lowest number of jumps needed to serve d(i,j) the microservice m(k)
        #   y (i,j,k,ii,jj) -> activation/desactivation variable to select the lowest posibility for z(i,j,k)   
        #   U               -> maximun value of Z
        for microservice in self.M:
            for drone in self.D:
                    self.model.add_var(f'x_{microservice},d{drone["position"]}', var_type=BINARY)
                    self.model.add_var(f'z_{microservice},d{drone["position"]}', var_type=CONTINUOUS, lb=0)
                    for drone2 in self.D:
                        self.model.add_var(f'y_{microservice},d({drone["position"]}),d({drone2["position"]})', var_type=BINARY)
        
        decision_variables = np.array(self.model.vars)
        U = self.model.add_var(f'U', var_type=CONTINUOUS, lb=0)


        # Add the drones limitations constraints
        #TODO The constraint must check against the limitations of the drones, not the deployment matrix. With this configuration a drone can hold a maximun of 1 microservices
        decision_variables_aux = decision_variables.reshape((len(self.M), len(self.D), -1))
        decision_variables_x = decision_variables_aux[:,:,0].reshape((len(self.M),-1)).T
        for i, x in enumerate(decision_variables_x):
            constraint = xsum(x) <= 1
            self.model.add_constr(constraint, name=f'Capacity of {self.D[i]["id"]} not surpased')


        # Add the mandatory deployment of all microservices constraint
        decision_variables_x = decision_variables_x.T
        for i, x in enumerate(decision_variables_x):
            constraint = xsum(x) >= 1
            self.model.add_constr(constraint, name=f'Microservice {self.M[i]} deployed')


        # Add the top boundaries for z (i,j,k) constraint to ensure tightness
        decision_variables_x = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,0]      
        decision_variables_z = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,1]
        decision_variables_y = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,2:].reshape((len(self.M), len(self.D), len(self.D)))


        for microservice in range(len(self.M)):
            for drone in range(len(self.D)):
                for drone2 in range(len(self.D)):
                    constraint = decision_variables_z[microservice][drone] >= self.C - ((decision_variables_x[microservice][drone2] * self.H[microservice][drone]) / self.P[drone][drone2]) - self.C * decision_variables_y[microservice][drone][drone2]
                    self.model.add_constr(constraint, name=f'Ensure Tightness of top boundary for  yz({self.M[microservice]}{self.D[drone]["id"]}{self.D[drone2]["id"]}) z({microservice}{self.D[drone]["id"]}{self.D[drone2]["id"]})')

        # Add the constraints that ensures that only one of the previous constraint is active for each z(k,i,j)
        for microservice in range(len(self.M)):
            for drone in range(len(self.D)):
                constraint = xsum([decision_variables_y[microservice][drone][drone2] for drone2 in range(len(self.D))]) == len(self.D) - 1
                self.model.add_constr(constraint, name=f'Check that only one top boundary constraint is active for y({microservice},{self.D[drone]["id"]},{self.D[drone2]["id"]})')

        for i, zetas in enumerate(decision_variables_z):
            for j,z in enumerate(zetas):
                constraint = U <= z 
                # self.model.add_constr(constraint, name=f'U greater than z{i},{j}')

        decision_variables_z = decision_variables.reshape((len(self.M), len(self.D), -1))[:,:,1]
        z_heats = np.array(self.H).reshape((-1))
        # Add the objetive function
        self.model.objective = minimize(xsum([z for zetas in decision_variables_z for z, z_heat in zip(zetas, z_heats)]))
        # self.model.objective = minimize(U)

    def addOptionalConstraints(self, nInstancesLimit):
        if (len(nInstancesLimit) != len(self.M)): return

        decision_variables = np.array(self.model.vars).reshape((len(self.M), self.T[0], self.T[1], 2))
        decision_variables = decision_variables[:,:,:,0].reshape((len(self.M), -1))
        # For each microservice, force the limit of instances to deploy

        for k in range(len(self.M)):
            constraint = xsum(decision_variables[k]) <= nInstancesLimit[k]
            self.model.add_constr(constraint, name=f'Ensure no more than {nInstancesLimit[k]} are deployed for ms {self.M[k]}')
                    
    def solve(self):

        status = self.model.optimize()
        print(status)

        df_data = [{'Decision Variable': var.name, 'Value': var.x} for index, var in enumerate(self.model.vars)]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
            df = pd.DataFrame(df_data)
        return pd.DataFrame(df_data)


    def flatten(self, xs):
        result = []
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(self.flatten(x))
        else:
            result.append(xs)
        return result
    
    def printSolution(self, background_img: str = None):
        solution_df = pd.DataFrame([{"Decision Variable": var.name, "Value": var.x} for var in self.model.vars[:-1]])

        solution_ndarray = solution_df.to_numpy()[:,1]
        solution_ndarray = solution_ndarray.reshape(len(self.M), len(self.D), -1)[:,:,0]
        deployment_matrix = np.zeros((len(self.M), self.T[0], self.T[1]))
        for microservice in range(len(self.M)):
            for i, drone in enumerate(self.D):
               deployment_matrix[microservice][drone['position'][0]][drone['position'][1]] =  solution_ndarray[microservice][i]


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
    
    def getMin(self, microservice_index, src_drone, proposed_solution):
        hops_to_dsts = []

        for dst_drone_index, dst_drone in enumerate(self.D):

                if (proposed_solution[microservice_index][dst_drone_index] == 1 ):
                    hops_to_dsts.append(self.P[src_drone['position'][0]*self.T[1]+src_drone['position'][1]][dst_drone['position'][0]*self.T[1]+dst_drone['position'][1]]-1 if self.P[src_drone['position'][0]*self.T[1]+src_drone['position'][1]][dst_drone['position'][0]*self.T[1]+dst_drone['position'][1]] != -1 else 999)
        return min(hops_to_dsts)

    def solutionLongFormat(self, scenario_name: str):

        proposed_solution = np.array([decision_variable.x for decision_variable in self.model.vars[:-1]])

        # print(len(proposed_solution))
        proposed_solution_aux = proposed_solution.reshape((len(self.M), len(self.D), -1))[:,:,0]
        print('Holita', len(proposed_solution_aux) ,proposed_solution_aux)
        # Calcular el número de saltos mínimo que hay que dar para, desde un dron de origen, poder comsumir un servicio
        # alojado en el dron de destino más cercano
        proposed_solution = []
        for ms in range(len(self.M)):
            for i ,d in enumerate(self.D): 
                if (self.H[ms][i] > 1):
                   proposed_solution.append(self.getMin(ms, d, proposed_solution_aux))
                else:
                    proposed_solution.append(-1)
                        
                       
        proposed_solution = np.array(proposed_solution).reshape((len(self.M), -1))
        proposed_solution = proposed_solution.tolist()
        
        for index, sublist in enumerate(proposed_solution):
            proposed_solution[index] = [element for element in sublist if element != -1]
          
        dataset = []
        for ms, sublist in zip(self.M, proposed_solution):
                for cost in sublist:
                    dataset.append([scenario_name, ms, cost])

        return dataset
        
    @staticmethod
    def compareSolutions(solutions_df):
        medianprops = {'linestyle':'-', 'linewidth':2, 'color':'lightcoral'}
        meanpointprops = {'marker':'D', 'markeredgecolor':'forestgreen','markerfacecolor':'forestgreen'}

        sns.set_theme(style="ticks", palette="pastel")


        fig, ax = plt.subplots(figsize=(20, 10))

        sns.boxplot(data= solutions_df, x="Microservice", y="Hops", hue="Scenario",
                    meanprops = meanpointprops, showmeans=True, 
                    flierprops={"marker": "x"},
                    medianprops=medianprops)

        plt.xticks(range(solutions_df['Microservice'].nunique()), solutions_df['Microservice'].unique(), rotation=45)
        # plt.yticks(range(0, int(df.loc[df['Hops'].idxmax()]['Hops'])+2))
        plt.yticks(range(0, 23))
        plt.xlabel('Microservices', fontsize=16)
        plt.ylabel('Latency (nº hops)', fontsize=16)
        plt.title('Medium sized Scenario', pad=30, fontsize=24)
        plt.legend(fontsize=14)
        plt.show()



configuration = json.load(open(r'./prueba_005.json'))

my_model = GLOMIP(configuration=configuration)
my_model.showDeploymentMatrix('scenario3.png')
my_model.showHeatMaps('scenario3.png')
my_model.initializeModel()
#my_model.addOptionalConstraints([100,100,100,100])
my_model.solve().to_csv('prueba_005_result.csv')
my_model.printSolution('scenario3.png')

scenarios = my_model.solutionLongFormat('Test_05')



configuration = json.load(open(r'./prueba_006.json'))

my_model = GLOMIP(configuration=configuration)
my_model.showHeatMaps('scenario3.png')
my_model.initializeModel()
#my_model.addOptionalConstraints([1,1,1,1])
my_model.solve().to_csv('prueba_006_result.csv')
my_model.printSolution('scenario3.png')

scenarios += my_model.solutionLongFormat('Test_06')

configuration = json.load(open(r'./prueba_007.json'))
my_model = GLOMIP(configuration=configuration)
my_model.showHeatMaps('scenario3.png')
#my_model.addOptionalConstraints([1,1,1,1])
my_model.initializeModel()
my_model.solve().to_csv('prueba_007_result.csv')
my_model.printSolution('scenario3.png')

scenarios += my_model.solutionLongFormat('Test_07')



df = pd.DataFrame(data = scenarios,
                  columns = ['Scenario','Microservice', 'Hops'])
print(df)
GLOMIP.compareSolutions(df)
