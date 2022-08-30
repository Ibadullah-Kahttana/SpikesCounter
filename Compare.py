
import csv
import os

name_of_test_image=''



with open('outputCSV\\Predicted_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count=0
    
    for row in csv_reader:
        
        if(count>=1):
            count2=0
            for r in row:
                if(count2==0):
                    name_of_test_image=r
                    print(name_of_test_image)
                    print(row)
                if(count2==1):
                    
                    a=''
                    counter=0
                    rowadd=''
                    os.remove("outputCSV\\Predicteddata.txt")
                    predict= open('Predicteddata.txt', 'a')

                    for t in range(len(r)):
                        if(r[t]!=' '):
                            a=a+r[t]
                        if(r[t]==' '):
                            print(a)
                            if(counter>1):
                                rowadd=rowadd+', '+a
                            elif(counter==1):
                                rowadd=rowadd+a
                            counter=counter+1
                            a=''
                        if(counter==5):
                            print(rowadd)
                            predict.write(rowadd+'\n')
                            rowadd=''
                            counter=0
                    rowadd=rowadd+', '+a
                    predict.write(rowadd)
                    predict.close() 
                    print(a)
                count2=count2+1
        count=count+1


            
matching_file=''
with open('TrainCSV\\train.csv') as train_csv:
    csv_reader = csv.reader(train_csv, delimiter=',')
    line_count = 0
    count=0
    matchcount=1
    for row in csv_reader:
        count=0
        for r in row:
            if(count==0): 
                matching_file=r
                if(matching_file==name_of_test_image):
                    string=''
                    rowadd=''
                    
                    counter=0
                    try:
                        os.remove("Actual.txt")
                    except:
                       abc=1
                    actual= open('Actual.txt', 'a')
                    print(row[3])
                    for n in row[3]:
                        if (n!=','):
                            string=string+n
                        if(n==' '):
                            print(string)
                            
                            if(counter>0):
                                rowadd=rowadd+', '+string
                            elif(counter==0):
                                rowadd=rowadd+string
                            
                            counter=counter+1
                            string=''
                        if(counter==3):
                            print(rowadd)
                            actual.write(rowadd+'\n')
                            rowadd=''
                            counter=0
                            
                    #rowadd=rowadd+', '+string
                    #print(rowadd)
                    #predict.write(rowadd)
                    
                    predict.close() 
                    print(a)
                    print (string)
                    print(row,matchcount)
                    matchcount=matchcount+1
                
            count=count+1
