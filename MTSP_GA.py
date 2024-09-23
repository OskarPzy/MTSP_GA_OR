import numpy as np
import copy
from matplotlib import pyplot as plt

#参数设定
spot_num=30               #生成的随机点数（社区数）
routespot_num=spot_num//5  #每条回路经过的社区数
entity_num=100            #每个种群中个体数
cross_prob=0.4            #基因交叉概率
mut_prob=0.02             #基因变异概率
GA_times=6000             #遗传次数

#生成随机点（社区）
def span_spots(spot_num):
    spots=[]
    for i in range(spot_num):
        x=np.random.randint(0,1000)
        y=np.random.randint(0,1000)
        spots.append(np.array([x,y]))
    spots.append(np.array([250,750]))
    spots.append(np.array([750,750]))
    spots.append(np.array([500, 500]))
    spots.append(np.array([250, 250]))
    spots.append(np.array([750, 250]))
    return np.array(spots)

#计算距离矩阵
def dis(spot1,spot2):
    dis=round(np.sqrt((spot1[0]-spot2[0])**2+(spot1[1]-spot2[1])**2),2)
    return dis
def span_DMAT(spots):
    length=len(spots)
    DMAT=np.zeros([length,length])
    for i in range(length):
        for j in range(length):
            DMAT[i,j]=dis(spots[i],spots[j])
    return DMAT

#初始化种群
def span_population(spot_num,entity_num):
    entity=list(range(spot_num))
    population=[]
    for i in range(entity_num):
        np.random.shuffle(entity)
        population.append(copy.deepcopy(entity))
    return population

#定义目标函数并计算适应度
def cut_routes(entity,spot_num,routespot_num):
    routes=[]
    for i in range(5):
        routes.append(entity[i*routespot_num:(i+1)*routespot_num])
    for i in range(5):
        routes[i].insert(0,spot_num+i)
        routes[i].append(spot_num+i)
    return routes

def aimfunction(routes,DMAT):
    total_len=0
    for route in routes:
        for i in range(len(route)-1):
            total_len += DMAT[route[i],route[i+1]]
    return total_len

def cal_fitness(population,spot_num,routespot_num,DMAT):
    fitness=[]
    for entity in population:
        routes=cut_routes(entity,spot_num,routespot_num)
        total_len=aimfunction(routes,DMAT)
        fitness.append(1.0/total_len)
    return fitness

#定义算子1——自然选择
def selection(population,fitness):
    #构建轮盘列表
    fitness_sum=[]
    for i in range(len(fitness)):
        if i==0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i-1]+fitness[i])
    for i in range(len(fitness_sum)):
        fitness_sum[i]/=sum(fitness)
    #轮盘选择
    population_new=[]
    for i in range(len(population)):
        rand=np.random.uniform(0,1)
        for j in range(len(fitness_sum)):
            if j==0:
                if 0<=rand and rand <= fitness_sum[j]:
                    population_new.append(population[j])
            else:
                if fitness_sum[j-1]<rand and rand<=fitness_sum[j]:
                    population_new.append(population[j])
    return population_new

#定义算子2——基因交叉（种群交配）
def amend(entity, low, high):                      #定义修正函数，防止交叉后基因重复
    #基因分段
    length = len(entity)
    cross_gene = entity[low:high]  # 交叉基因
    raw = entity[0:low] + entity[high:]  # 非交叉基因
    not_in_cross = []  # 不应交叉基因
    for i in range(length):
        if i not in cross_gene:
            not_in_cross.append(i)
    #检错并修正
    error_index = []
    for i in range(len(raw)):
        if raw[i] in not_in_cross:
            not_in_cross.remove(raw[i])
        else:
            error_index.append(i)
    for i in range(len(error_index)):
        raw[error_index[i]] = not_in_cross[i]
    #构建修正后的基因型
    new_entity = raw[0:low] + cross_gene + raw[low:]
    return new_entity

def crossover(population_new, cross_prob):                 #定义基因交叉函数
    half = int(len(population_new) / 2)
    father = population_new[:half]
    mother = population_new[half:]
    np.random.shuffle(father)
    np.random.shuffle(mother)
    offspring = []
    for i in range(half):
        if np.random.uniform(0, 1) <= cross_prob:
            cut1 = np.random.randint(0, 9)
            cut2= cut1+20
            son = father[i][0:cut1] + mother[i][cut1:cut2] + father[i][cut2:]
            son = amend(son, cut1, cut2)
            daughter = mother[i][0:cut1] + father[i][cut1:cut2] + mother[i][cut2:]
            daughter = amend(daughter, cut1, cut2)
        else:
            son = father[i]
            daughter = mother[i]
        offspring.append(son)
        offspring.append(daughter)
    return offspring

#定义算子3——基因变异
def mutation(offspring, mut_prob):
    for i in range(len(offspring)):
        if np.random.uniform(0, 1) <= mut_prob:
            position1 = np.random.randint(0, len(offspring[i]))
            position2 = np.random.randint(0, len(offspring[i]))
            offspring[i][position1],offspring[i][position2] = offspring[i][position2],offspring[i][position1]
    return offspring

#主程序
#构建点集并图像输出
spots=np.load("数据集\spots_data.npy")
DMAT=np.load("数据集\DMAT_data.npy")
plt.scatter(spots[:spot_num,0],spots[:spot_num,1],c='b')
plt.scatter(spots[spot_num:,0],spots[spot_num:,1],c='r')
plt.savefig("spots_fig.png")
#运行算法10次，取其最优
min_len=10000
op_population=[]
op_process=[]
for time in range(10):
    # 遗传算法求解
    population = span_population(spot_num, entity_num)  # 初始化种群
    op_sol = []
    for i in range(GA_times + 1):  # 遗传迭代
        old_population = copy.deepcopy(population)
        # 自然选择
        fitness = cal_fitness(population, spot_num, routespot_num, DMAT)
        old_fitness = copy.deepcopy(fitness)
        population_new = selection(population, fitness)
        # 基因交叉
        offspring = crossover(population_new, cross_prob)
        # 基因变异
        population = mutation(offspring, mut_prob)
        fitness = cal_fitness(population, spot_num, routespot_num, DMAT)
        if sum(fitness) < sum(old_fitness):
            population = old_population
            fitness = old_fitness
        # 收敛数据采集
        if i % 100 == 0:
            # 找到该种群中最优个体/最优解
            fitness = cal_fitness(population, spot_num, routespot_num, DMAT)
            op_entity = population[fitness.index(max(fitness))]
            op_routes = cut_routes(op_entity, spot_num, routespot_num)
            op_sol.append(aimfunction(op_routes, DMAT))

    #更新全局最优解
    if op_sol[-1]<min_len:
        min_len=op_sol[-1]
        op_population=population
        op_process=op_sol

#输出最终最优结果
fitness = cal_fitness(op_population, spot_num, routespot_num, DMAT)
op_entity = op_population[fitness.index(max(fitness))]
op_routes = cut_routes(op_entity, spot_num, routespot_num)
col_ls = ['green', 'yellow', 'pink', 'orange', 'purple']
plt.figure(1)
for i in range(len(op_routes)):
    plt.plot(spots[op_routes[i], 0], spots[op_routes[i], 1], c=col_ls[i])
plt.scatter(spots[:spot_num, 0], spots[:spot_num, 1], c='b')
plt.scatter(spots[spot_num:, 0], spots[spot_num:, 1], c='r')
plt.savefig("路径规划.png")
plt.figure(2)
plt.plot(op_process, c='b')
plt.savefig("收敛过程.png")
#命令行输出
print("点集：")
print(spots)
print("最优解={}".format(op_entity))
print('最短路径={}'.format(min_len))
plt.show()