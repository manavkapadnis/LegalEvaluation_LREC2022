import graphicalModel 

no=1
for i in range(93):
  graphicalModel.get_summary('./data/'+str(no)+'.txt',no)
  no+=1