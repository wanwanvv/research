import os
ori_betweenness_fileName= open("./betweenness.values", "r") #original graph
overlay_betweenness_fileName= open("./betweenness.values", "r") #overlay graph
nodeMapping_fileName= open("./newTodelete.txt", "w+") #overlay graph
#list
ori_bet={}
over_bet={}
ori_node_list = []
node_map = {}
new_id_rank = {}
#读取txt文件中每一行的值
for line in ori_betweenness_fileName:
    items = line.strip("\n").split(" ")
    id=int(items[0])
    id_w=int(items[1])
    ori_bet[id]=id_w
    ori_node_list.append(id)
ori_betweenness_fileName.close()
rank=0
for line in overlay_betweenness_fileName:
    items = line.strip("\n").split(" ")
    id=int(items[0])
    id_w=int(items[1])
    over_bet[id]=id_w
    new_id_rank[id]=rank
    rank+ = 1
overlay_betweenness_fileName.close()
new_id=0
for line in nodeMapping_fileName:
    items = line.strip("\n")
    old_id=int(items[0])
    node_map[old_id]=new_id
    new_id+=1
nodeMapping_fileName.close()
write_fileName=open("./Betweenness.compare", "w+")
old_rank=0
new_rank=0
for old_id in ori_node_list:
    write_fileName.write(str(old_rank)+"-"+str(old_id)+"-"+str(ori_bet[old_id]))
    if over_bet.__contains__(old_id):
        new_id = node_map[old_id]
        new-rank = new_id_rank[new_id]
        write_fileName.write(" "+str(new_rank)+"-"+str(new_id)+"-"+str(over_bet[new_id]))
        if ori_bet[old_id] == over_bet[new_id]:
            write_fileName.write(" 1")
        else:
            write_fileName.write(" 0")
    write_fileName.write("\n")
    old_rank+=1
write_fileName.close()

