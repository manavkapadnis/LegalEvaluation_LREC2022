import letsum 
import letsum_test
import csv

no=1
for i in range(93):
  letsum_test.LetSum('./data/'+str(no)+'.txt',no)
  no+=1