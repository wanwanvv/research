import os
ori_betweenness_fileName= open("./betweenness.values", "r") #original graph
overlay_betweenness_fileName= open("./betweenness.values", "r") #overlay graph
nodeMapping_fileName= open("./newTodelete.txt", "r") #overlay graph
#list
ori_bet={}
over_bet={}
#读取txt文件中每一行的值
for line in open(ori_betweenness_fileName, "r"):
#每个输入数据以空格隔开
    items = line.strip("\n").split(" ")
    id=int(items[0])
    id_w=int(items[1])
    ori_bet[id]=id_w
ori_betweenness_fileName.close()
for line in open(overlay_betweenness_fileName, "r"):
    items = line.strip("\n").split(" ")
    id=int(items[0])
    id_w=int(items[1])
    over_bet[id]=id_w
overlay_betweenness_fileName.close()
write_fileName=open("./Betweenness.compare", "w+")
new_id=0
for line in open(nodeMapping_fileName, "r"):
    items = line.strip("\n")
    old_id=int(items[0])
    write_fileName.write(str(old_id)+"-"+str(ori_bet[old_id])+" "+str(new_id)+"-"+str(over_bet[new_id]))
    if ori_bet[old_id]== over_bet[new_id]:
        write_fileName.write("1")
    else:
        write_fileName.write("0")
    write_fileName.write("\n")
    new_id+=1
nodeMapping_fileName.close()
write_fileName.close()

