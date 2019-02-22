## Click-Through-Rate Prediction with PySpark Logistic Regression


Pyspark for Implementing Linear Logistic Regression and Polynomial Logistic Regression
This implementation of Logistic Regression is for Click-Through-Rate prediction!



> This well documented report was designed to either run in an _Amazon Elastic Map Reduce_ cluster or to run on a container service with pyspark on a local computer.
The Pyspark implementation for Logistic regression in this workbook uses RDDs (resilient distributed datasets)
The project starts with data exploration using the MLLib library and pySparks native RDDs.

### Exploratory Data Analysis

> The exploratory data analysis phase was essential for this project since we needed to develop a Logistic regression model 
that regressed over 200 independent variables on the Click vs no-Click dependent variable. 
The number of characteristics that were eventually available include:
- Total number of NAN values
- Total number of unique features
- The mean, max, standard deviation of all the numeric features


### Hashing of the Categorical Independent Variables

> The categorical variables were hashed using a new hashing algorithm that maps the hashing value to the relevance of the count occurence of the value to the predictability of a click (Ground Truth).
By evaluating all categorical variables together using a universal hashing function, the hashing becomes much more applicable to a wider array of data and can be amenable also to streaming data.

The code using the native pyspark RDD is shown below

```
StartFromA = 14
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
    
        """ The Hashing Function converts all categorical variables based on feature importance to values between 
            1 and 10"""
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
    
    YY = TrainRDD.map(ValuesNonNumericFeatures)\
                .reduceByKey(lambda x,y: x+y)\
                .cache()

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
        

    """ The dictionary containing feature importance is broadcasted to all partitions for use in conversion 
        of the categorical variables to features with lower cardinality - denser representation"""  
    
    YYY = sc.broadcast(NamingDict)
    
    
    def MappingChangesWithDictionary(trainRDD):


        """ Mapping Changes with What the Dictionary has for feature importance """
        Dictionary = YYY.value

        """ Taking in all the Key/Value Components """
        FinalKey = trainRDD[0]
        Value = trainRDD[1]
        Value[Node] = Dictionary.get(Value[Node], 0)

        return (FinalKey,Value)



    NonNumericRDD = NonNumericRDD.map(MappingChangesWithDictionary)\
                    .cache()
    
    
NonNumericRDD.take(2)
```


### Home-Grown Logistic Regresion Algorithm

> A custom built logistic regression algorithm was also written using pySparks native RDDs. 
To ensure that computation could be done on EMR clusters(in AWS) as well as on a local computer, we ensured that the one-hot-encoding of the categorical variables occured only during computation thus moving one to computation (memory limiting) intensive programming rather than a more space/hardware limitation.

> The logistic regression algorithm was used for prediction using only the 14 independent numerical variables and also using both the numerical and non-numerical independent variables. The accuracy, as expected, was better when all the independent variables were considered.


### Model Evaluation

> The perofmance of the logistic regression model was analysed using accuracy, precision, recall as well as F1-score metrics.
They were all home-grown and computed during each epoch.

> Standard gradient descent was utilized, although one could argue that a faster approach would have been to utilize stochastic gradient descent or batch gradient descent.

> As a bonus _Polynomial Logistic Regression Algorithm_ was also written for this project although there was no time to implement the algorithm on the Click Through Rate prediction.


