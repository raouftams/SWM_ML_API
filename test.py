from utilities import openPkl, savePkl


data = openPkl('models/year_cluster.pkl')
print(data)

#for item in data.items():
#    if(item[1] == 3 or item[1] == 1):
#        data[item[0]] = 0
#    else:
#        data[item[0]] = 1

data['C031'] = 2
savePkl(data, 'models/year_cluster.pkl')
#data = openPkl('models/week_cluster.pkl')
#print(data)