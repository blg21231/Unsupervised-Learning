#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import random
from array import array
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from random import sample


# ## Summary

# In[2]:


# Create Tic Tac Toe simulator

# Simulations v1
#     Simulate games using randomStrategy to select best spots
# Logistic Regression v1
#     Train logistic regression model using Simulations v1 data
    
# Simulations v2
#     Simulate games using randomStrategy vs logReStrategy1
# Logistic Regression v2
#     Train logistic regression model using Simulations v2 data
    
# Simulations v3
#     Simulate games using logReStrategy1 vs logReStrategy2
# Logistic Regression v3
#     Train logistic regression model using Simulations v3 data
    
# Simulations v4
#     Simulate games using logReStrategy3 vs logReStrategy4
# Logistic Regression v4
#     Train logistic regression model using Simulations v4 data


# ## Board

# In[3]:


boardLength = 3
boardWidth = 3


# ## Methods

# In[4]:


def createBoard(boardWidth, boardLength):
    board = pd.DataFrame(columns = np.arange(boardWidth))
    for l in range(boardLength):
        board.loc[len(board.index)] = [0]*boardWidth
    return board


# In[5]:


def countScore(mark):
    longestRow = 0
    longestCol = 0
    longestDiaR = 0
    longestDiaL = 0
    
    currentRow = 0
    currentCol = 0
    currentDiaR = 0
    longestDiaL = 0
    
    previousRow = 0
    previousCol = 0
    previousDiaR = 0
    previousDiaL = 0
    
    #longest row
    for r in range(boardLength):
        for c in range(boardWidth):
            if c == 0: 
                currentRow = 0
            if board.loc[r,c] != mark:
                currentRow = 0
            else:
                if previousRow == mark:
                    currentRow += 1
                else:
                    currentRow = 1
                longestRow = max(longestRow, currentRow)
            previousRow = board.loc[r,c]
            
    #longest col
    for c in range(boardWidth):
        for r in range(boardLength):
            if r == 0:
                currentCol = 0
            if board.loc[r,c] != mark:
                currentCol = 0
            else:
                if previousCol == mark:
                    currentCol += 1
                else:
                    currentCol = 1
                longestCol = max(longestCol, currentCol)
            previousCol = board.loc[r,c]

    #longest dia \
    for r in range(boardLength):
        for c in range(boardWidth):
            if r == 0 or c == 0:
                for d in range(min(boardLength - r, boardWidth - c)):
                    if board.loc[r+d,c+d] != mark:
                        currentDiaR = 0
                    else:
                        if previousDiaR == mark:
                            currentDiaR += 1
                        else:
                            currentDiaR = 1
                        longestDiaR = max(longestDiaR, currentDiaR)
                    previousDiaR = board.loc[r+d,c+d]
                    
    #longest dia /
    for r in range(boardLength):
        for c in range(boardWidth):
            if r == 0 or c == boardWidth - 1:
                for d in range(min(boardLength - r, c+1)):
                    if board.loc[r+d,c-d] != mark:
                        currentDiaL = 0
                    else:
                        if previousDiaL == mark:
                            currentDiaL += 1
                        else:
                            currentDiaL = 1
                        longestDiaL = max(longestDiaL, currentDiaL)
                    previousDiaL = board.loc[r+d,c-d]
            
    return max(longestRow, longestCol, longestDiaR, longestDiaL)


# ## Test

# In[6]:


board = createBoard(boardLength, boardWidth)
board.loc[0,0] = 1
board.loc[1,1] = 1
board.loc[1,2] = -1
board.loc[2,1] = -1
board


# In[7]:


countScore(1)


# In[8]:


countScore(-1)


# ## Strategies

# In[9]:


def randomStrategy():
    spot = [random.randint(0,len(board.index) - 1), random.randint(0,len(board.columns) - 1)]
    while board.loc[spot[0], spot[1]] != 0:
        spot = [random.randint(0,len(board.index) - 1), random.randint(0,len(board.columns) - 1)]
    return spot


# In[10]:


# for each spot on the board
# check if that spot is available
# if so, predict the probability of winning and losing after marking that spot
# if newWinChance:newLoseChance odds r higher than current best odds, set it to bestWinChance
# return bestSpot

def logReStrategy1(mark):
    bestScore = 0
    bestSpot = []
    available = []
    for r in range(len(board.index)):
        for c in range(len(board.columns)):
            if board.loc[r,c] == 0:
                newBoard = board.copy()
                newBoard.loc[r,c] = mark
                
                newScore = np.dot(logRe1.predict_proba([newBoard.to_numpy().flatten()]).flatten(), logRe1.classes_.tolist())

                if newScore > bestScore:
                    bestScore = newScore
                    bestSpot = [r,c]
                else:
                    available.append([r,c])
    if bestSpot == []:
        bestSpot = sample(available,1)
    return bestSpot


# In[11]:


# for each spot on the board
# check if that spot is available
# if so, predict the probability of winning and losing after marking that spot
# if newWinChance:newLoseChance odds r higher than current best odds, set it to bestWinChance
# return bestSpot

def logReStrategy2(mark):
    bestScore = 0
    bestSpot = []
    available = []
    for r in range(len(board.index)):
        for c in range(len(board.columns)):
            if board.loc[r,c] == 0:
                newBoard = board.copy()
                newBoard.loc[r,c] = mark
                
                newScore = np.dot(logRe1.predict_proba([newBoard.to_numpy().flatten()]).flatten(), logRe1.classes_.tolist())

                if newScore > bestScore:
                    bestScore = newScore
                    bestSpot = [r,c]
                else:
                    available.append([r,c])
    if bestSpot == []:
        bestSpot = sample(available,1)
    return bestSpot


# ## Simulations v1: using randomStrategy predictions

# In[12]:


# use randomStrategy to select best spot
# mark spot
# update moves
# after game ends, go back and mark outcomes
# repeat for 1000 runs

moves = []
scores = []
outcomes = []
priorNumMoves = 0

for i in range(1000):
    board = createBoard(boardWidth, boardLength)
    spotsRemaining = boardWidth * boardLength
    
    currentMark = [-1,1][random.randint(0,1)]
    
    x_score = 0
    o_score = 0
    
    while max(x_score, o_score) < min(boardWidth, boardLength) and (spotsRemaining > 0):
        spot = randomStrategy()
        board.loc[spot[0], spot[1]] = currentMark
        moves.append(board.to_numpy().flatten())
        spotsRemaining -= 1
        
        if currentMark == 1:
            x_score = countScore(currentMark)
        elif currentMark == -1:
            o_score = countScore(currentMark)
            
        scores.append(x_score - o_score)
        currentMark *= -1
            
    if x_score >= min(boardWidth, boardLength):
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(1)
    elif o_score >= min(boardWidth, boardLength):
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(-1)
    else:
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(0)
            
    priorNumMoves = len(moves)


# In[13]:


spotsRemaining


# In[14]:


board


# In[15]:


moves


# In[16]:


outcomes


# In[17]:


len(moves) == len(outcomes)


# ## Data Cleaning

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(moves, scores, test_size=0.20)


# ## Logistic Regression v1: training on Simulations v1 data

# In[19]:


logRe1 = LogisticRegression().fit(X_train, y_train)


# In[20]:


logRe1.C 


# In[21]:


predictions = logRe1.predict(X_test)


# In[22]:


logRe1.C 


# In[23]:


correctPredictions = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correctPredictions += 1


# In[24]:


correctPredictions/len(predictions)


# In[25]:


frequency = {}
# iterating over the list
for item in y_test:
   # checking the element in dictionary
   if item in frequency:
      # incrementing the counr
      frequency[item] += 1
   else:
      # initializing the count
      frequency[item] = 1

# printing the frequency
print(frequency)


# In[26]:


print("logReStrategy{} predicts with {}% accuracy".format(1, round(correctPredictions/len(predictions)*100,1)))


# ## Simulations v2: using logReStrategy1 predictions

# In[27]:


# randomly pick who goes first
# use logReStrategy1 to select best spot for each player on their turn
# mark spot
# update moves
# after game ends, go back and mark outcomes
# partial fit logRe1 model
# repeat for 1000 runs

moves = []
scores = []
outcomes = []
priorNumMoves = 0
priorReTrain = 0

for i in range(1000):
    board = createBoard(boardWidth, boardLength)
    spotsRemaining = boardWidth * boardLength
    
    currentMark = [-1,1][random.randint(0,1)]
    
    x_score = 0
    o_score = 0
        
    while max(x_score, o_score) < min(boardWidth, boardLength) and (spotsRemaining > 0):
        spot = logReStrategy1(currentMark)
        board.loc[spot[0], spot[1]] = currentMark
        moves.append(board.to_numpy().flatten())
        spotsRemaining -= 1
        
        if currentMark == 1:
            x_score = countScore(currentMark)
        elif currentMark == -1:
            o_score = countScore(currentMark)
            
        scores.append(x_score - o_score)
        currentMark *= -1
            
    if x_score >= min(boardWidth, boardLength):
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(1)
    elif o_score >= min(boardWidth, boardLength):
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(-1)
    else:
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(0)
            
    logRe1 = LogisticRegression(warm_start = True).fit(moves[priorNumMoves:], scores[priorNumMoves:])        
    priorNumMoves = len(moves)


# In[ ]:


spotsRemaining


# In[ ]:


board


# In[ ]:


moves


# In[ ]:


outcomes


# In[ ]:


len(moves) == len(outcomes)


# ## Data Cleaning

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(moves, scores, test_size=0.20)


# ## Logistic Regression v2: training on Simulations v2 data

# In[ ]:


logRe2 = LogisticRegression().fit(X_train, y_train)


# In[ ]:


predictions = logRe2.predict(X_test)


# In[ ]:


correctPredictions = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correctPredictions += 1


# In[ ]:


correctPredictions/len(predictions)


# In[ ]:


frequency = {}
# iterating over the list
for item in y_test:
   # checking the element in dictionary
   if item in frequency:
      # incrementing the counr
      frequency[item] += 1
   else:
      # initializing the count
      frequency[item] = 1

# printing the frequency
print(frequency)


# In[ ]:


print("logReStrategy{} predicts with {}% accuracy".format(2, round(correctPredictions/len(predictions)*100,1)))


# ## Simulations v3: using logReStrategy2 predictions

# In[ ]:


# randomly pick who goes first
# use logReStrategy2 to select best spot for each player on their turn
# mark spot
# update moves
# after game ends, go back and mark outcomes
# partial fit logRe2 model
# repeat for 1000 runs

moves = []
scores = []
outcomes = []
priorNumMoves = 0
priorRetrain = 0

for i in range(1000):
    board = createBoard(boardWidth, boardLength)
    spotsRemaining = boardWidth * boardLength
    
    currentMark = [-1,1][random.randint(0,1)]
    
    x_score = 0
    o_score = 0
        
    while max(x_score, o_score) < min(boardWidth, boardLength) and (spotsRemaining > 0):
        spot = logReStrategy2(currentMark)
        board.loc[spot[0], spot[1]] = currentMark
        moves.append(board.to_numpy().flatten())
        spotsRemaining -= 1
        
        if currentMark == 1:
            x_score = countScore(currentMark)
        elif currentMark == -1:
            o_score = countScore(currentMark)
            
        scores.append(x_score - o_score)
        currentMark *= -1
            
    if x_score >= min(boardWidth, boardLength):
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(1)
    elif o_score >= min(boardWidth, boardLength):
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(-1)
    else:
        for i in range(len(moves)-priorNumMoves):
            outcomes.append(0)
            
    logRe2 = LogisticRegression(warm_start = True).fit(moves[priorNumMoves:], outcomes[priorNumMoves:])
    priorNumMoves = len(moves)


# In[ ]:


spotsRemaining


# In[ ]:


board


# In[ ]:


moves


# In[ ]:


outcomes


# In[ ]:


len(moves) == len(outcomes)


# ## Data Cleaning

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(moves, scores, test_size=0.20)


# ## Logistic Regression v3: training on Simulations v3 data

# In[ ]:


logRe3 = LogisticRegression().fit(X_train, y_train)


# In[ ]:


predictions = logRe3.predict(X_test)


# In[ ]:


correctPredictions = 0
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        correctPredictions += 1


# In[ ]:


correctPredictions/len(predictions)


# In[ ]:


frequency = {}
# iterating over the list
for item in y_test:
   # checking the element in dictionary
   if item in frequency:
      # incrementing the counr
      frequency[item] += 1
   else:
      # initializing the count
      frequency[item] = 1

# printing the frequency
print(frequency)


# In[ ]:


print("logReStrategy{} predicts with {}% accuracy".format(3, round(correctPredictions/len(predictions)*100,1)))

