import matplotlib.pyplot as plt
import random
from scipy.signal import lti, step2
import numpy as np
from control.matlab import *
import math

DNA_SIZE        = 3
POPULATION_SIZE = 1000
GENERATIONS     = 10
MUTATION_PROB   = 0.05
MIN_RANDOM      = 0.01
MAX_RANDOM      = 20

def step_info(t,yout):
    chars = []
    chars.append(((yout.max()/yout[-1]-1)*100))
    for i in xrange(len(t)):
        if(abs(y[-1-i] - y[-1])/y[-1] > 0.02 ):
            chars.append(t[-i])
            break;
    return chars
    
def randomGain():
    return random.uniform(MIN_RANDOM, MAX_RANDOM)

def randomPop():
    pop = []
    for i in xrange(POPULATION_SIZE):
        dna = []
        for c in xrange(DNA_SIZE):
            dna.append(randomGain())
        pop.append(dna)
    return pop

def findITAE(reference, y,t):
    error = []
    for i in range(len(y)):
        error.append(abs((reference - y[i])*t[i])*t[99]/100.0)
    return sum(error)

def findISE(reference, y,t):
    error = []
    for i in range(len(y)):
        error.append((reference - y[i])*(reference - y[i])*t[99]/100.0)
    return sum(error)

def findIAE(reference, y,t):
    error = []
    for i in range(len(y)):
        error.append(abs((reference-y)[i])*t[99]/100.0)
    return sum(error)

def weightedChoice(items):
    weightTotal = sum((item[1] for item in items))
    n = random.uniform(0, weightTotal)
    for item, weight in items:
        if n < weight:
            return item
        n = n - weight
    return item

def crossOver(dna1, dna2):
    pos = int(random.random() * DNA_SIZE)
    return (dna1[:pos]+dna2[pos:], dna2[:pos]+dna1[pos:])

def mutate(dna):
    dnaOut = []
    for c in xrange(DNA_SIZE):
        if random.uniform(0, 1) <= MUTATION_PROB:
            dnaOut.append(randomGain()) # Mutated
        else:
            dnaOut.append(dna[c])
    return dnaOut


if __name__ == "__main__":
    # Setpoint
    reference = 1

    # Plant parameter
    a = -5.89; b = 95.49; c = 19.87; d = 186.3

    # Random initial population
    population = randomPop()

    # Best individual intialization
    bestFitnessInd = population[0]
    bestPopFitness = 0
    avg = []

    # Loop every generation
    for generation in xrange(GENERATIONS):
        print "Generation %s... Random sample: '%s'" % (generation, population[0])
        weightedPop = []
        averageFit = 0

        # Individual loop in every population
        for individual in population:

            # Find individual step response
            Kp = individual[0]; Ki = individual[1]; Kd = individual[2]/40;
            sys = tf([a, b],[1, c, d]) * tf([Kd, Kp, Ki],[1, 0])
            f = feedback(sys)
            y,t = step(f)
            
            # Find individual fitness
            fitnessVal = 1.0/(1.0+findITAE(reference,y,t)) # Bisa diubah IAE, ISE, atau ITAE
            if (math.isnan(fitnessVal) == True):
                fitnessVal = 0
            elif (y[0] < -0.1):
                fitnessVal *= 0.8
            
            # Find average fitness of the generation
            averageFit += fitnessVal

            # Record best individual through generations
            if(fitnessVal > bestPopFitness): 
                bestFitnessInd = individual
                bestPopFitness = fitnessVal 
            pair = [individual, fitnessVal]
            
            # plt.clf()
            # plt.plot(t,y)
            # plt.title(str(fitnessVal))
            # plt.pause(0.00001)
            
            # Population with fitness calculation on every individual
            weightedPop.append(pair)
        # Calculate population average fitness
        averageFit /= POPULATION_SIZE
        avg.append(averageFit)
        print "Generation %s average fitness is %s " % (generation, averageFit)
        
        # Create new population for the next generation
        population = []

        # Crossover and mutation
        for _ in xrange(POPULATION_SIZE/2):
            ind1 = weightedChoice(weightedPop)
            ind2 = weightedChoice(weightedPop)

            ind1, ind2 = crossOver(ind1, ind2)

            population.append(mutate(ind1))
            population.append(mutate(ind2))
    
    
    

    # plt.show('hold')

    # Showing the best individual over generations
    print "%s\t%s\t%s"%(bestFitnessInd[0], bestFitnessInd[1], bestFitnessInd[2])
    Kp = bestFitnessInd[0]; Ki = bestFitnessInd[1]; Kd = bestFitnessInd[2]/40;
    sysb = tf([a, b],[1, c, d])
    fb = feedback(sysb)
    yb,tb = step(fb)
    

    sys = tf([a, b],[1, c, d]) * tf([Kd, Kp, Ki],[1, 0])
    f = feedback(sys)
    y,t = step(f)
    plt.plot(tb,yb,'r', label='Without PID')
    plt.plot(t,y,'b', label='With PID')
    plt.legend()
    plt.show()
    print bestPopFitness
    print step_info(t,y)
    for i in xrange(GENERATIONS):
        print avg[i]

