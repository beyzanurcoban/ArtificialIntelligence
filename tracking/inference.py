# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import random
import re
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        totalValues = self.total()

        if(len(self.values()) != 0 and totalValues != 0): # If list is not empty and If all the values are not 0
            for key,value in self.items():
                self[key] = value/totalValues
        

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        # First normalize the distribution, then sort it according to the values
        # Then, produce a random number between 0 and 1
        # Check the interval that this number corresponds
        # If the number is in that interval, return the key of that list element
        # Else, move to the next interval

        """
        If sortedList = [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        Then, sortedList[0] = ('a', 0.2) and sortedList[0][0] = 'a' and sortedList[0][1] = 0.2
        This list properties are used in the for loop to find the appropriate interval
        """
    
        self.normalize()
        sortedList = list(sorted(self.items()))
        rnd = random.random()

        valueSum = 0
        length = len(self.values())

        for i in range(length):
            if rnd < valueSum+sortedList[i][1]:
                return sortedList[i][0]
            else:
                valueSum += sortedList[i][1]


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        distancePG = manhattanDistance(pacmanPosition, ghostPosition)

        # P(the ghost is in jail, and we detect it) = 1
        if(ghostPosition == jailPosition and noisyDistance is None):
            return 1
        # P(the ghost is in jail, but we couldn't detect it) = 0
        elif(ghostPosition == jailPosition and noisyDistance is not None):
            return 0
        # P(the ghost is not in jail, but we interpret it as in jail) = 0
        elif(ghostPosition != jailPosition and noisyDistance is None):
            return 0
        
        return busters.getObservationProbability(noisyDistance, distancePG)


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # B(X_t) = P(E_t|X_t)B'(X_t) where B(X_t) = P(X_t|e1:t) and B'(X_t) = P(X_t|e1:t-1) 

        allPos = self.allPositions

        for pos in allPos:
            # Get P(E_t|X_t)
            obsProbability = self.getObservationProb(observation, gameState.getPacmanPosition(), pos, self.getJailPosition())

            # Get B'(X_t)
            if(self.beliefs.get(pos) is None):
                b_old = 0
            else:
                b_old = self.beliefs.get(pos)

            # B(X_t) = P(E_t|X_t)B'(X_t)
            b_new = obsProbability * b_old

            self.beliefs.update({pos:b_new})

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # P(X_t+1) = sum over X_t [P(X_t+1|X_t)P(X_t)]

        allPos = self.allPositions
        newDistribution = DiscreteDistribution()

        for pos in allPos:
            newPosDist = self.getPositionDistribution(gameState, pos)

            # Get P(X_t)
            if(self.beliefs.get(pos) is None):
                b_old = 0
            else:
                b_old = self.beliefs.get(pos)

            # Get P(X_t+1|X_t)
            for dist in newPosDist:
                if(newPosDist.get(dist) is None):
                    transitionProb = 0
                else:
                    transitionProb = newPosDist.get(dist)
                
                # Sum over X_t
                if(newDistribution.get(dist) is None):
                    nextStep = 0
                else:
                    nextStep = newDistribution.get(dist)

                # P(X_t+1) = sum over X_t [P(X_t+1|X_t)P(X_t)]
                newDistribution.update({dist: transitionProb*b_old + nextStep})
                    
        self.beliefs = newDistribution.copy()
        self.beliefs.normalize()


    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        positions = self.legalPositions 

        # We will place self.numParticles many particles, so the for loop does this.
        # We will evenly distribute the particles, so I implemented a round between the
        # elements of the positions list. When we reach at the end of the list, we go back
        # to the first element by modulo operation.
        for num in range(self.numParticles):
            self.particles.append(positions[num%len(positions)])


    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # w(x) = P(e|x)
        newParticleDist = DiscreteDistribution()
        allParticles = self.particles

        for particle in allParticles:
            # Get the observation probability
            w = self.getObservationProb(observation, gameState.getPacmanPosition(), particle, self.getJailPosition)
            
            # Get the old belief
            if newParticleDist.get(particle) is None:
                b_old = 0
            else:
                b_old = newParticleDist.get(particle)

            # Update the distribution entry
            newParticleDist.update({particle:b_old+w})

        # If all values are zero, then we couldn't get a valid distribution. We cannot get any info from this distribution.
        # We create a uniform initial distribution. Then, we get beliefs.
        # Otherwise, just normalize what you found.
        if newParticleDist.total()==0:
            self.initializeUniformly(gameState)
            newParticleDist = self.getBeliefDistribution()
        else:
            newParticleDist.normalize()

        # Resample from the updated weights
        newParticles = []
        for j in range(self.numParticles):
            newParticles.append(newParticleDist.sample())

        self.particles = newParticles
        

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        allPos = self.allPositions
        newDistribution = DiscreteDistribution()
        beliefs = self.getBeliefDistribution()

        for pos in allPos:
            newPosDist = self.getPositionDistribution(gameState, pos)

            # Get P(X_t)
            if(beliefs.get(pos) is None):
                b_old = 0
            else:
                b_old = beliefs.get(pos)

            # Get P(X_t+1|X_t)
            for dist in newPosDist:
                if(newPosDist.get(dist) is None):
                    transitionProb = 0
                else:
                    transitionProb = newPosDist.get(dist)
                
                # Sum over X_t
                if(newDistribution.get(dist) is None):
                    nextStep = 0
                else:
                    nextStep = newDistribution.get(dist)

                # P(X_t+1) = sum over X_t [P(X_t+1|X_t)P(X_t)]
                newDistribution.update({dist: transitionProb*b_old + nextStep})
                    
        samples = [newDistribution.sample() for _ in range(int(self.numParticles))]
        self.particles = samples


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        beliefs = DiscreteDistribution()
        for particle in self.particles:
            # For every particle, increment the belief by 1
            beliefs[particle] += 1

        beliefs.normalize()
        return beliefs


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        positions = self.legalPositions 
        
        particles = list(itertools.product(positions, repeat=self.numGhosts))

        for num in range(self.numParticles):
            self.particles += particles
            num += len(particles)

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        newParticleDist = DiscreteDistribution()
        allParticles = self.particles

        for particle in allParticles:
            # Get the observation probability
            obsProb = 1

            # For all ghosts, find the distribution. Multiply them. 
            # This will give the total distribution.
            for i in range(self.numGhosts):
                noisyGhostObs = observation[i]
                obsProb = obsProb * self.getObservationProb(noisyGhostObs, gameState.getPacmanPosition(), particle[i], self.getJailPosition(i))
            
            # Get the old belief
            if newParticleDist.get(particle) is None:
                b_old = 0
            else:
                b_old = newParticleDist.get(particle)

            # Update the distribution entry
            newParticleDist.update({particle:b_old+obsProb})

        # If all values are zero, then we couldn't get a valid distribution. We cannot get any info from this distribution.
        # We create a uniform initial distribution. Then, we get beliefs.
        # Otherwise, just normalize what you found.
        if newParticleDist.total()==0:
            self.initializeUniformly(gameState)
            newParticleDist = self.getBeliefDistribution()
        else:
            newParticleDist.normalize()

        # Resample from the updated weights
        newParticles = []
        for j in range(self.numParticles):
            newParticles.append(newParticleDist.sample())

        self.particles = newParticles


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"      
            # # For all ghosts, find the distribution.
            # for ghost in range(self.numGhosts):
            #     ghostDist = self.getPositionDistribution(gameState, oldParticle, ghost, self.ghostAgents[ghost])
            #     # Resample the particle by just looking at the new disribution of 1 ghost.
            #     newParticlePos = ghostDist.sample()
            #     # Update the enry of newParticle
            #     newParticle[ghost] = newParticlePos   
            raiseNotDefined()        
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
