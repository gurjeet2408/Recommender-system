import csv

try:
    import numpy
except:
    print("This implementation requires the numpy module.")
    exit(0)

def Exact_Matcher(A,B):
    Matrix = [[0 for x in range(len(A))] for y in range(len(B))]
    for i in range(len(A)):
        for j in range(len(B)):
            if (A[i].lower()==B[j].lower()):
                Matrix[i][j]=1
            else:
                Matrix[i][j]=0
    return(Matrix)

def Partial_Matcher(A,B):
    Matrix = [[0 for x in range(len(A))] for y in range(len(B))]
    count=0
    for i in range(len(A)):
        for j in range(len(B)):
            count=0
            str1=row1[i].lower()
            str2=row2[j].lower()
            if(str2.find(str1)!=-1):
                count=count+1
            if(str1.find(str2)!=-1):
                count=count+1

            Matrix[i][j]=count
            Matrix[i][j]=(Matrix[i][j])/2
    return Matrix
def Data_Checker(A,B):
    Matrix = [[0 for x in range(len(A))] for y in range(len(B))]
    for i in range (len(A)):
        for j in range(len(B)):
            if A[i]==B[j]:
                Matrix[i][j]=1
            else:
                Matrix[i][j]=0
    return Matrix
def loadWords(file):
    list = [] # create an empty list to hold the file contents
    #file_contents = codecs.open(file, "r", "utf-8") # open the file
    for line in file: # loop over the lines in the file
        line = line.strip() # strip the line breaks and any extra spaces
        list.append(line) # append the word to the list
    #print(list)
    return list

def lev_dist(source, target):
    if source == target:
        return 0

    # Prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in range(slen+1):
        dist[i][0] = i
    for j in range(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1,   # deletion
                            dist[i+1][j] + 1,   # insertion
                            dist[i][j] + cost   # substitution
                        )
    return dist[-1][-1]

def Matrix_Factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T
if __name__ == "__main__":
    #Source Dataset as 'qld_data.csv'
    with open('qld_person.csv', 'r') as f:
        reader = csv.reader(f)
        row1 = next(reader)
        row3=next(reader)
        #print(row1)
    #Target Dataset as 'VISTA_HTS_Data2.csv'
    with open('VISTA_HTS_Data2.csv','r') as h:
        reader = csv.reader(h)
        row2=next(reader)
        row4=next(reader)
        #print(row2)
    list1 = loadWords(row1)
    list2 = loadWords(row2)
    w, h= len(row1), len(row2)
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(list1)):       # so now you are looping over a range of numbers, not lines
        for j in range(len(list2)):
            Matrix[i][j]= lev_dist(list1[i],list2[j])
            Matrix[i][j]=23-Matrix[i][j]
            #print(lev_dist(list1[i], list2[j]))
    print("Lavenshtein",Matrix)
    #M1=max(max(Matrix))
    #for i in range (len(Matrix)):
        #Matrix[i]= M1-Matrix[i]
    #print(Matrix)
    E1= Exact_Matcher(row1,row2)
    print("ExactMatcher",E1)
    E2= Partial_Matcher(row1,row2)
    print("Partial Matcher",E2)
    E3= Data_Checker(row3,row4)
    print("Data Checker",E3)
    result1 = [[0 for x in range(len(E1))] for y in range(len(E2))]
    result2 = [[0 for x in range(len(E3))] for y in range(len(Matrix))]
    result3 = [[0 for x in range(len(result1))] for y in range(len(result2))]
    for i in range(len(E1)):
        for j in range(len(E2)):
            result1[i][j]=E1[i][j]+E2[i][j]
            result2[i][j]=E3[i][j]+Matrix[i][j]
            result3[i][j]=result1[i][j]+result2[i][j]



    #print(result1)
    #print(result2)
    print(result3)

    result3 = numpy.array(result3)
    N = len(result3)
    M = len(result3[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = Matrix_Factorization(result3, P, Q, K)
    print(nP)
    print(nQ)
