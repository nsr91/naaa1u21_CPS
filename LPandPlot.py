from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd
import matplotlib.pyplot as mPLT
import numpy as np

#The plotting function which will draw the bar graphs
def plot(model, count):
    hours = [str(x) for x in range(0, 24)]
    pstn = np.arange(len(hours)) #postion
    users = ['user1', 'user2', 'user3', 'user4', 'user5'] #users
    color_list = ['darkred','darkorange','darkgreen','khaki','azure']
    plot_list = []
    to_plot = []

    #Making the plot lists to plot the energy usage
    for user in users:
        t_list = []
        for hour in hours:
            hour_list_temp = []
            task_count = 0
            for vr in model.variables():
                if user == vr.name.split('_')[0] and str(hour) == vr.name.split('_')[2]:
                    task_count += 1
                    hour_list_temp.append(vr.value())
            t_list.append(sum(hour_list_temp))
        plot_list.append(t_list)

    #Creating bar charts for every user
    mPLT.bar(pstn,plot_list[0],color=color_list[0],edgecolor='darkblue',bottom=0)
    mPLT.bar(pstn,plot_list[1],color=color_list[1],edgecolor='darkblue',bottom=np.array(plot_list[0]))
    mPLT.bar(pstn,plot_list[2],color=color_list[2],edgecolor='darkblue',bottom=np.array(plot_list[0])+np.array(plot_list[1]))
    mPLT.bar(pstn,plot_list[3],color=color_list[3],edgecolor='darkblue',bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2]))
    mPLT.bar(pstn,plot_list[4],color=color_list[4],edgecolor='darkblue',bottom=np.array(plot_list[0])+np.array(plot_list[1])+np.array(plot_list[2])+np.array(plot_list[3]))

    mPLT.xticks(pstn, hours)
    mPLT.xlabel('Time in hours')
    mPLT.ylabel('The Energy Usage [kW]')
    mPLT.title('The 5 Users Energy Usage In 24 hours \nDay:  %i'%count)
    mPLT.legend(users,loc=0)
    mPLT.savefig('plot/plot\\'+str(count)+'.png') #saving the draws as .png  (plot#.png)
    mPLT.clf()

    return plot_list

#Getting the users tasks details from the Exel file COMP3217CW2Input.xlsx
excelFile = pd.read_excel ('COMP3217CW2Input.xlsx', sheet_name = 'User & Task ID')
task_name = excelFile['User & Task ID'].tolist()
ready_time = excelFile['Ready Time'].tolist()
deadline = excelFile['Deadline'].tolist()
max_energy_hour = excelFile['Maximum scheduled energy per hour'].tolist()
energy_demand = excelFile['Energy Demand'].tolist()
tasks = []
task_names = []

for r in range (len(ready_time)):
    task = []
    task.append(ready_time[r])
    task.append(deadline[r])
    task.append(max_energy_hour [r])
    task.append(energy_demand [r])
    task_names.append(task_name[r])
    tasks.append(task)

#Reading TestingResults.txt data
testData = pd.read_csv('TestingResults.txt', header=None)
labels_y = testData[24].tolist()
testData = testData.drop(24, axis=0)
data_x = testData.values.tolist()

#Scheduleing and plotting the  abnormal curves
for index, price_list in enumerate(data_x):
    if labels_y[index] == 1:
        #temperary variables
        task_t = []
        c = []
        ap = []

        #LP model minimize defintion
        model = LpProblem(name="scheduling-problem", sense=LpMinimize)

        #go through the tasks
        for index, task in enumerate(tasks):
            n = task[1] - task[0] + 1
            t_list = []
            for i in range(task[0], task[1] + 1):
                x = LpVariable(name=task_names[index]+'_'+str(i), lowBound=0, upBound=task[2])
                t_list.append(x)
            task_t.append(t_list)

        #construct the objective function and put it in the model
        for index, task in enumerate(tasks):
            for vr in task_t[index]:
                price = price_list[int(vr.name.split('_')[2])]
                c.append(price * vr)
        model += lpSum(c)

        #Add additional constraints to the model
        for index, task in enumerate(tasks):
            t_list = []
            for vr in task_t[index]:
                t_list.append(vr)
            ap.append(t_list)
            model += lpSum(t_list) == task[3]

    ans = model.solve()
    print(ans)
    plot(model,index+1)  #plotting 24 hour energy usage for 5 users using the scheduling code
print("\n Done!! all the abnormal curves has been created and plotted")
