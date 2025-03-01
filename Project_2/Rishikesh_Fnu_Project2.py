import numpy as np
import matplotlib.pyplot as plt

nTrain = 10
xTrain = np.random.uniform(0,1,nTrain)
tTrain = np.ones((nTrain,1))
tTrain[:,0] = np.sin(2*np.pi*xTrain) + np.random.normal(0,0.3,nTrain)
# plt.scatter(xTrain, tTrain)
# plt.show()

nTest = 100
xTest = np.random.uniform(0,1,nTest)
tTest = np.ones((nTest,1))
tTest[:,0] = np.sin(2*np.pi*xTest) + np.random.normal(0,0.3,nTest)
# plt.scatter(xTest, tTest)
# plt.show()


errTrain = []
errTest = []
print("N Train = 10: ")
for m in range(0,10):
    tempList = []
    tempListTest = []
    for i in range(m+1):
        tempList.insert(0, xTrain**i)
        tempListTest.insert(0, xTest**i)
    phi = np.array(tempList)
    phiTest = np.array(tempListTest)
    phi = np.transpose(phi)
    phiTest = np.transpose(phiTest)
    phiP = np.linalg.pinv(phi)
    w = phiP.dot(tTrain)
    print(w)

    x0 = np.linspace(0,1)
    y = 0
    temp = m
    for i in range(len(w)):
        y+= w[i]*x0**m
        m = m-1
    m = temp
    # plt.plot(x0, y, color="purple", label='Closed Form')
    # plt.scatter(xTrain, tTrain, color="orange")
    # plt.scatter(xTest, tTest, color="pink")
    # plt.xlim(0,1)
    # plt.ylim(-1.5,1.5)
    # plt.show()

    errTrain.append(np.sqrt((np.linalg.norm(phi.dot(w)-tTrain, ord =2)**2)/nTrain))
    errTest.append((np.sqrt((np.linalg.norm(phiTest.dot(w)-tTest, ord =2)**2)/nTest)))
    print("\nTraining error for M = {} is:".format(m))
    print(errTrain[m])
    print("Testing error for M = {} is:".format(m))
    print(errTest[m])

plt.plot(errTrain, label = "Training")
plt.plot(errTest, label = "Testing")
plt.title("N = 10")
plt.legend()
# plt.plot((errTrain - min(errTrain)) / (max(errTrain) - min(errTrain)))
# plt.plot((errTest - min(errTest)) / (max(errTest) - min(errTest)))
plt.xlim(0,10)
plt.xlabel("M")
plt.ylim(0,1)
plt.ylabel("$\mathregular{E_{RMS}}$")
plt.show()

#######################################################################################


nTrain = 100
xTrain = np.random.uniform(0,1,nTrain)
tTrain = np.ones((nTrain,1))
tTrain[:,0] = np.sin(2*np.pi*xTrain) + np.random.normal(0,0.3,nTrain)
# plt.scatter(xTrain, tTrain)
# plt.show()

nTest = 100
xTest = np.random.uniform(0,1,nTest)
tTest = np.ones((nTest,1))
tTest[:,0] = np.sin(2*np.pi*xTest) + np.random.normal(0,0.3,nTest)
# plt.scatter(xTest, tTest)
# plt.show()


errTrain = []
errTest = []
print("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ")
print("\n\nN Train = 100: ")
for m in range(0,10):
    tempList = []
    tempListTest = []
    for i in range(m+1):
        tempList.insert(0, xTrain**i)
        tempListTest.insert(0, xTest**i)
    phi = np.array(tempList)
    phiTest = np.array(tempListTest)
    phi = np.transpose(phi)
    phiTest = np.transpose(phiTest)
    phiP = np.linalg.pinv(phi)
    w = phiP.dot(tTrain)

    x0 = np.linspace(0,1)
    y = 0
    temp = m
    for i in range(len(w)):
        y+= w[i]*x0**m
        m = m-1
    m = temp
    # plt.plot(x0, y, color="purple", label='Closed Form')
    # plt.scatter(xTrain, tTrain, color="orange")
    # plt.scatter(xTest, tTest, color="pink")
    # plt.xlim(0,1)
    # plt.ylim(-1.5,1.5)
    # plt.show()

    errTrain.append(np.sqrt((np.linalg.norm(phi.dot(w)-tTrain, ord =2)**2)/nTrain))
    errTest.append((np.sqrt((np.linalg.norm(phiTest.dot(w)-tTest, ord =2)**2)/nTest)))
    print("\nTraining error for M = {} is:".format(m))
    print(errTrain[m])
    print("Testing error for M = {} is:".format(m))
    print(errTest[m])

plt.plot(errTrain, label = "Training")
plt.plot(errTest, label = "Testing")
plt.title("N = 100")
plt.legend()
# plt.plot((errTrain - min(errTrain)) / (max(errTrain) - min(errTrain)))
# plt.plot((errTest - min(errTest)) / (max(errTest) - min(errTest)))
plt.xlim(0,10)
plt.xlabel("M")
plt.ylim(0,1)
plt.ylabel("$\mathregular{E_{RMS}}$")
plt.show()