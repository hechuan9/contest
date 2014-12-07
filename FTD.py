# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(CaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def __init__( self, index, timeForComputing = .1 ):
    CaptureAgent.__init__(self, index, timeForComputing)

    self.discount = 0.9
    self.noise = 0
    self.alpha = 0.01
    self.epsilon = 0.05

    self.preScore = 0
    self.preState = None
    self.preAction = None
    self.weights = util.Counter()

    self.weights = util.Counter()
    self.weights['distanceToFood'] = -1
    self.weights['successorScore'] = 100
    #self.weights['onOffence'] = 0
    #self.weights['numDefenders'] = 0
    #self.weights['defenderDistance'] = 0
    #self.weights['attackerDistance'] = 0
    self.weights['foodsLeft'] = -1
    self.weights['foodsRemained'] = 1
    self.weights = util.normalize(self.weights)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    score = len(self.getFood(gameState).asList())
    reward = (score - self.preScore) * 50
    self.update(reward, gameState)

    actions = gameState.getLegalActions(self.index)

    if util.flipCoin(self.epsilon): #exploration
      action = random.choice(actions)
    else:
      action = self.getPolicy(gameState, actions)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    self.preState = gameState
    self.preAction = action
    self.preScore = score
    return action

  def getPolicy(self, gameState, actions):
    if len(actions) == 0:
      return None
    q_values = []
    for action in actions:
      q_values.append(self.getQValue(gameState, action))
    index = q_values.index(max(q_values))
    return actions[index]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features['successorScore'] = self.getScore(successor)
    features['onOffence'] = 0
    if myState.isPacman: features['onOffence'] = 1

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    myPos = successor.getAgentState(self.index).getPosition()
    features['foodsLeft'] = len(foodList)
    minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
    features['distanceToFood'] = minDistance

    friends = [successor.getAgentState(i) for i in self.getTeam(successor)]
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    attackers = [a for a in friends if a.isPacman and a.getPosition() != None]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    #features['numDefenders'] = len(defenders)

    if len(attackers) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in attackers]
      #features['attackerDistance'] = max(dists)
    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      #features['defenderDistance'] = min(dists)

    return features

  def getWeights(self, gameState, action):
    return self.weights

  def update(self, reward, gameState):
    if self.preState == None:
      return
    if not gameState.getAgentState(self.index).isPacman: #don't update unless it is in offense position
      return

    correction = (reward + self.discount * self.getValue(gameState)) - self.getQValue(self.preState, self.preAction)
    #print correction
    features = self.getFeatures(self.preState, self.preAction)
    for feature in features:
      #print feature
      #print self.weights[feature]
      self.weights[feature] = self.weights[feature] + self.alpha * correction * features[feature]

    self.weights = util.normalize(self.weights)
    print self.weights

  def getValue(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    actions = gameState.getLegalActions(self.index)
    if len(actions) == 0:
      return 0
    q_values = []
    for action in actions:
      q_values.append(self.getQValue(gameState, action))
    return max(q_values)

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    qValue = 0
    features = self.getFeatures(gameState, action)
    for feature in features:
      qValue += self.weights[feature] * features[feature]
    return qValue

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def __init__( self, index, timeForComputing = .1 ):
    ReflexCaptureAgent.__init__(self, index, timeForComputing)

    self.weights = util.Counter()
    self.weights['numInvaders'] = -1000
    self.weights['onDefense'] = 100
    self.weights['invaderDistance'] = -10
    self.weights['stop'] = -100
    self.weights['reverse'] = -2
    #self.weights = util.normalize(self.weights)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return self.weights
