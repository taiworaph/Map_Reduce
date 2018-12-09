# [START pyspark]
import pyspark

import re
import ast
import time
import numpy as np
import pandas as pd


sc = pyspark.SparkContext()

trainRDD = sc.textFile("gs://raphael2/train.txt")


## Precomputed Feature Means and Standard Deviations for mean normalization of numerical variables ##

TrainRDD, Train2RDD = trainRDD.randomSplit([0.7,0.5], seed = 2018)
TrainRDD.cache()
Train2RDD.unpersist()


def NonNumericValuesOnly(LineRDD):
    """Function takes the RDD and subsets the RDD"""
    Values = LineRDD.split('\t')
    ZZ = []
    RealKey = str(Values[14]) + str(Values[15]) + str(Values[16])
    for ii in range(14,40):
        if Values[ii] == '':
            ZZ.append(str(0))
        else:
            ZZ.append(str(Values[ii]))
    
    return (RealKey, ZZ)


def MakingRDDDataFrames(rowRDD):
    
    """Getting the RDD in the format of Sets and Lists"""
    Key = rowRDD[0]
    ####################
    
    Values = rowRDD[1]
    
    ### Getting each individual Value WoW
    Str14 = Values[0]
    Str15 = Values[1]
    Str16 = Values[3]
    Str17 = Values[4]
    Str18 = Values[5]
    Str19 = Values[6]
    Str20 = Values[7]
    Str21 = Values[8]
    Str22 = Values[9]
    Str23 = Values[10]
    Str24 = Values[11]
    Str25 = Values[12]
    Str26 = Values[13]
    Str27 = Values[14]
    Str28 = Values[15]
    Str29 = Values[16]
    Str30 = Values[17]
    Str31 = Values[18]
    Str32 = Values[19]
    Str33 = Values[20]
    Str34 = Values[21]
    Str35 = Values[22]
    Str36 = Values[23]
    Str37 = Values[24]
    Str38 = Values[25]
    Str39 = Values[26]
    
    return(FinalKey, Str14, Str15, Str16, Str17, Str18, Str19, Str20, Str21, Str22, Str23, Str24, Str25, Str26, Str27, Str28, Str29,
          Str30, Str31, Str32, Str33, Str34, Str35, Str36, Str37, Str38, Str39)


NonNumericRDD = TrainRDD.map(NonNumericValuesOnly).cache()

count =0
for ii in range(14, 40):
    Node = ii
    def ValuesNonNumericFeatures(toyRDDLine):

        """ Take the node value from a broadcast variable that is sent to the function """
        Values = toyRDDLine.split('\t')
        T14 = Values[Node]
        return (T14, 1)


    def NonNumericFeatures(toyRDDLine):

        """ Take the node value from a broadcast variable that is sent to the function """
        Values = toyRDDLine.split('\t')
        T14 = Values[Node]
        T0 = Values[0]
        return (((T14, T0), 1))

    def ConvertForMerge(toyRDDLine):

        YY = toyRDDLine[0][0]
        YYY = toyRDDLine[0][1]
        YYZ = toyRDDLine[1]

        return ((YY,(YYY,YYZ)))


    def MovingOn(trainRDD):

        Key = trainRDD[0]
        Value = trainRDD[1]
        
        ValueKey = Value[0][0]
        ValueValue = Value[0][1]
        ValueDivide = Value[1]
        
        return ((Key, (ValueKey, float(ValueValue/ValueDivide))))


    def HasherFunction(trainRDD):
    
        Key = trainRDD[0]
        Value = trainRDD[1][1]
        
        if (Value >0.9):
            FinalValue = 1
        elif (Value <=0.9) & (Value >0.8):
            FinalValue = 2
        elif (Value <=0.8) & (Value >0.7):
            FinalValue = 3
        elif (Value <=0.7) & (Value>0.6):
            FinalValue = 4
        elif (Value <=0.6) & (Value >0.5):
            FinalValue = 5
        elif (Value <=0.5) & (Value >0.4):
            FinalValue = 6
        elif (Value <=0.4) & (Value >0.3):
            FinalValue = 7
        elif (Value <=0.3) & (Value >0.2):
            FinalValue =8
        elif (Value <=0.2) & (Value >0.1):
            FinalValue = 9
        elif (Value <= 0.1):
            FinalValue = 10
            
        return ((Key, FinalValue))


    
    


#def ConvertToSignificance(toyRDDLine):
    

#print(trainRDD.map(ValuesNonNumericFeatures).reduceByKey(lambda x,y: x+y).takeOrdered(100,lambda x: -x[1]))
    
    YY = TrainRDD.map(ValuesNonNumericFeatures).reduceByKey(lambda x,y: x+y).cache()

    ZZ = TrainRDD.map(NonNumericFeatures)\
                 .reduceByKey(lambda x,y: x+y)\
                 .map(ConvertForMerge)\
                 .filter(lambda x: x[1][0] == '1')\
                 .leftOuterJoin(YY)\
                 .map(MovingOn).map(HasherFunction).cache()
    TT = ZZ.collect()

    ZZ.unpersist()
    YY.unpersist()

    #NamingDict = "HashDictionary" + str(Node)
    from collections import defaultdict

    NamingDict = defaultdict(list)

    for ii in TT:
        NamingDict[ii[0]] = ii[1]
        
        
    Node = count
    if count == 0:
        
        YYY = sc.broadcast(NamingDict)
        def MappingChangesWithDictionary(trainRDD):


            """ Mapping Changes with What the Dictionary Has """
            Dictionary = YYY.value
    
            """ Taking in all the Key/Value Components """
            FinalKey = trainRDD[0]
            
            
            
            Value = trainRDD[1]
            Values = Dictionary.get(Value[Node], 0)
            
            return (FinalKey,[Values])


        Hashing_1 = NonNumericRDD.map(MappingChangesWithDictionary).cache()
    
    else:
        
        YYY.unpersist()
        YYY = sc.broadcast(NamingDict)
        def MappingChangesWithDictionary(trainRDD):


            """ Mapping Changes with What the Dictionary Has """
            Dictionary = YYY.value
    
            """ Taking in all the Key/Value Components """
            FinalKey = trainRDD[0]
            
            
            
            Value = trainRDD[1]
            Values = Dictionary.get(Value[Node], 0)
            
            return (FinalKey,[Values])


        Hashing_2 = NonNumericRDD.map(MappingChangesWithDictionary).cache()
        Hashing_1 = Hashing_1.leftOuterJoin(Hashing_2).cache()
    count+=1
Hashing_1.take(200)


