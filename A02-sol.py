##from utils import distance_squared, turn_heading
from statistics import mean
##from ipythonblocks import BlockGrid
#from IPython.display import HTML, display
from time import sleep

import random
import copy
import collections

import numpy as np
from random import shuffle
from random import randrange


# ______________________________________________________________________________


class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        # Do we need this?
        pass


class Agent(Thing):
    """An Agent is a subclass of Thing with one required slot,
    .program, which should hold a function that takes one argument, the
    percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method. If it were a method,
    then the program could 'cheat' and look at aspects of the agent.
    It's not supposed to do that: the program can only look at the
    percepts. An agent program that needs a model of the world (and of
    the agent itself) will have to build and maintain its own model.
    There is an optional slot, .performance, which is a number giving
    the performance measure of the agent in its environment."""

    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0
        if program is None or not isinstance(program, collections.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(
                self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

        self.program = program

    def can_grab(self, thing):
        """Return True if this agent can grab this thing.
        Override for appropriate subclasses of Agent and Thing."""
        return False


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action
    agent.program = new_program
    return agent

# ______________________________________________________________________________

# ______________________________________________________________________________

'''
def SimpleReflexAgentProgram(rules, interpret_input):
    """This agent takes action based solely on the percept. [Figure 2.10]"""
    def program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        action = rule.action
        return action
    return program


def ModelBasedReflexAgentProgram(rules, update_state, model):
    """This agent takes action based on the percept and state. [Figure 2.12]"""
    def program(percept):
        program.state = update_state(program.state, program.action, percept, model)
        rule = rule_match(program.state, rules)
        action = rule.action
        return action
    program.state = program.action = None
    return program


def rule_match(state, rules):
    """Find the first rule that matches state."""
    for rule in rules:
        if rule.matches(state):
            return rule
'''
# ______________________________________________________________________________


loc_A, loc_B = (0, 0), (1, 0)  # The two locations for the Vacuum world

def ReflexVacuumAgent():
    """A reflex agent for the two-state vacuum environment. [Figure 2.8]
    >>> agent = ReflexVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """
    def program(percept):
        location, status = percept
        if status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'
    return Agent(program)


def ModelBasedVacuumAgent():
    """An agent that keeps track of what locations are clean or dirty.
    >>> agent = ModelBasedVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """
    model = {loc_A: None, loc_B: None}

    def program(percept):
        """Same as ReflexVacuumAgent, except if everything is clean, do NoOp."""
        location, status = percept
        model[location] = status  # Update the model here
        if model[loc_A] == model[loc_B] == 'Clean':
            return 'NoOp'
        elif status == 'Dirty':
            return 'Suck'
        elif location == loc_A:
            return 'Right'
        elif location == loc_B:
            return 'Left'
    return Agent(program)

# ______________________________________________________________________________


class Environment():
    """Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []  # List of classes that can go into environment

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def is_done(self):
        """By default, we're done when we can't find a live agent."""
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive:
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)
            self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            if self.is_done():
                return
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)"""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            thing.location = location if location is not None else self.default_location(thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class Direction:
    """A direction class for agents that want to move in a 2D plane
        Usage:
            d = Direction("down")
            To change directions:
            d = d + "right" or d = d + Direction.R #Both do the same thing
            Note that the argument to __add__ must be a string and not a Direction object.
            Also, it (the argument) can only be right or left."""

    R = "right"
    L = "left"
    U = "up"
    D = "down"
    '''
    NoOp = "no operation"
    rR = "rotate right"
    rL = "rotate left"
    '''

    def __init__(self, direction):
        self.direction = direction

    def __add__(self, heading):
        """
        >>> d = Direction('right')
        >>> l1 = d.__add__(Direction.L)
        >>> l2 = d.__add__(Direction.R)
        >>> l1.direction
        'up'
        >>> l2.direction
        'down'
        >>> d = Direction('down')
        >>> l1 = d.__add__('right')
        >>> l2 = d.__add__('left')
        >>> l1.direction == Direction.L
        True
        >>> l2.direction == Direction.R
        True
        """
        if self.direction == self.R:
            return{
                self.R: Direction(self.D),
                self.L: Direction(self.U),
            }.get(heading, None)
        elif self.direction == self.L:
            return{
                self.R: Direction(self.U),
                self.L: Direction(self.D),
            }.get(heading, None)
        elif self.direction == self.U:
            return{
                self.R: Direction(self.R),
                self.L: Direction(self.L),
            }.get(heading, None)
        elif self.direction == self.D:
            return{
                self.R: Direction(self.L),
                self.L: Direction(self.R),
            }.get(heading, None)

    def move_forward(self, from_location):
        """
        >>> d = Direction('up')
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (0, -1)
        >>> d = Direction(Direction.R)
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (1, 0)
        """
        x, y = from_location
        if self.direction == self.R:
            return (x + 1, y)
        elif self.direction == self.L:
            return (x - 1, y)
        elif self.direction == self.U:
            return (x, y - 1)
        elif self.direction == self.D:
            return (x, y + 1)

##-------------------------------------------------------------------------##
        
def SimpleReflexAgentProgram(rules, interpret_input):
    """This agent takes action based solely on the percept. [Figure 2.10]"""
    def program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        action = rule.action
        return action
    return program


def ModelBasedReflexAgentProgram(rules, update_state, model):
    """This agent takes action based on the percept and state. [Figure 2.12]"""
    def program(percept):
        program.state = update_state(program.state, program.action, percept, model)
        rule = rule_match(program.state, rules)
        action = rule.action
        return action
    program.state = program.action = None
    return program


def rule_match(state, rules):
    """Find the first rule that matches state."""
    for rule in rules:
        if rule.matches(state):
            return rule




class MavEnvironment(Environment):
    def __init__(self,things = [],agents = []):
        self.grid = Grid()
        self.matching_Grid = Grid()
        self.agent = MavSimpleReflexAgent()
        self.temp_Grid = Grid()
        self.things = []
        self.agents = []

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def check_Orientation(self,puzz_Grid,orig_Grid):
        if np.array_equal(puzz_Grid,orig_Grid):
            return 1
        return 0

    def result(self):
        if isinstance(self.agent,MavSimpleReflexAgent):
            print("Reflex : No of correct pieces: ",self.agent.get_moves(),"/",self.agent.grid_size * self.agent.grid_size," No of moves utilized: ",self.agent.get_total_moves())
        else: 
            print("Model : No of correct pieces: ",self.agent.get_moves(),"/",self.agent.grid_size * self.agent.grid_size," No of moves utilized: ",self.agent.get_total_moves())

        
    def get_Cordinates(self,cordinates):
        #print(self.grid_size,"-------->>>",self.grid_size - 1)
        cords = self.agent.get_Location(cordinates)
        while self.agent.if_Puzzle_Visited(cords):
            if cords == (self.agent.grid_size - 1,self.agent.grid_size - 1):
                for i in range(0,self.agent.grid_size):
                    print(self.temp_Grid.get_Grid()[i])
                
                return cords
            if self.puzzle_Check():
               
                break
            cords = self.agent.get_Location(cordinates)
        return cords
    
    def puzzle_Check(self):
        count = 0
        cmp_grid = np.identity(self.agent.get_Grid_Size(), dtype = float)
        for i in range(0,self.agent.get_Grid_Size()):
             for j in range(0,self.agent.get_Grid_Size()):
                 if np.array_equal(self.temp_Grid.get_Grid()[i][j],self.matching_Grid.get_Grid()[i][j]):
                      count += 1
                 else:
                     return 0
        return 1
                

    def total_correct_pieces(self):
        for i in range(0,self.agent.get_Grid_Size()):
             for j in range(0,self.agent.get_Grid_Size()):
                 if np.array_equal(self.temp_Grid.get_Grid()[i][j],self.matching_Grid.get_Grid()[i][j]):
                     self.agent.add_move()

                     
                 
        
    def execute_action(self,stepCount):
        match_Grid = Grid()
        match_Grid.set_Grid(self.grid.get_Grid())
        self.temp_Grid.set_Grid(self.grid.get_Grid())
        tile = []
        if isinstance(self.agent,MavSimpleReflexAgent):
            cordinates = self.agent.get_Random_Location()
            percept = (cordinates, self.check_Orientation(self.temp_Grid.get_Grid()[cordinates[0]][cordinates[1]],self.matching_Grid.get_Grid()[cordinates[0]][cordinates[1]]))
            for _ in range(0,stepCount):
                self.agent.add_total_move()
                if self.agent.program(percept):
                    #print("Already Sorted---------------->>>",cordinates)
                    cordinates = self.agent.get_Cordinates(cordinates)
                else:
                    i,j = cordinates
                    tile = self.temp_Grid.get_Grid()[i][j]
                    while self.agent.program(percept) == 0:
                        #print(tile)
                        #print(self.matching_Grid.get_Grid()[i][j])
                        tile = self.temp_Grid.rotate_antiClockwise(tile)
                        #print(tile)
                        #self.set_Grid(self.grid)
                        percept = (cordinates, self.check_Orientation(tile,self.matching_Grid.get_Grid()[i][j]))
                    self.temp_Grid.get_Grid()[i][j] = copy.deepcopy(tile)
                    tile2 = np.identity(2, dtype = float)
                    match_Grid.get_Grid()[i][j] = copy.deepcopy(tile2)
                    if self.puzzle_Check():
                        self.result()
                        break 
                    cordinates = self.agent.get_Cordinates(cordinates)
                percept = (cordinates, self.check_Orientation(self.temp_Grid.get_Grid()[cordinates[0]][cordinates[1]],self.matching_Grid.get_Grid()[cordinates[0]][cordinates[1]]))
           
        if isinstance(self.agent,MavModelBasedAgent):
           
            cordinates = self.agent.get_Random_Location()
            cordinates = self.agent.calcualte_referance_Coordinates(cordinates)
            percept = (cordinates, self.check_Orientation(self.temp_Grid.get_Grid()[cordinates[0]][cordinates[1]],self.matching_Grid.get_Grid()[cordinates[0]][cordinates[1]]))
            for _ in range(0,stepCount):
                self.agent.add_total_move()
                if self.agent.program(percept):
                   
                    self.agent.puzzle_Visied(cordinates)
                    cordinates = self.get_Cordinates(cordinates)
                else:
                    i,j = cordinates
                    tile = self.temp_Grid.get_Grid()[i][j]
                    while self.agent.program(percept) == 0:
                        tile = self.temp_Grid.rotate_antiClockwise(tile)
                        percept = (cordinates, self.check_Orientation(tile,self.matching_Grid.get_Grid()[i][j]))
                    self.temp_Grid.get_Grid()[i][j] = copy.deepcopy(tile)
                    tile2 = np.identity(2, dtype = float)
                    match_Grid.get_Grid()[i][j] = copy.deepcopy(tile2)
                    self.agent.puzzle_Visied(cordinates)
                    cordinates = self.get_Cordinates(cordinates)
                    if self.puzzle_Check():
                       
                        return
                percept = (cordinates, self.check_Orientation(self.temp_Grid.get_Grid()[cordinates[0]][cordinates[1]],self.matching_Grid.get_Grid()[cordinates[0]][cordinates[1]]))
           


    def add_thing(self,thing):
        if not isinstance(thing, Grid):
            self.agent = thing
        else:
            self.grid = thing
            
    def set_Matching_Grid(self,Grid):
        self.matching_Grid = Grid

    def set_Grid(self,thing):
        self.grid = copy.deepcopy(thing)
        self.temp_Grid = copy.deepcopy(thing)
        
    def delete_thing(self,thing):
        super().delete_thing(thing)

    def print_things(self):
        print('\n',self.things,'\n')
    

class MavDirection:
    def __init__(self):
        self.i = 0
        self.j = 0
        
    def update_Cordinates(self,cordinates,motion,size):
        self.i,self.j = cordinates
        if motion == 0 or motion == 1 or motion == 2 :
            if self.i==0:
                if self.j == size:
                    self.j-= 1
                elif self.j==0:
                    self.j+= 1
                else:
                    if randrange(0,2):
                        self.j += 1
                    else:
                        self.j -= 1
            else:
                self.i-= 1
        if motion == 3 or motion == 4 or motion == 5:
            if self.i==size:
                if self.j == size:
                    self.j-= 1
                elif self.j==0:
                    self.j+= 1
                else:
                    if randrange(0,2):
                
                        self.j += 1
                    else:
                
                        self.j -= 1
            else:
               
                self.i+= 1
        if motion == 6 or motion == 7 or motion == 8:
            if self.j==0:
                if self.i == size:
                    self.i-= 1
                elif self.i==0:
                    self.i+= 1
                else:
                    if randrange(0,2):
                
                        self.i += 1
                    else:
                 
                        self.i -= 1
            else:
             
                self.j-= 1
        if motion == 9 or motion == 10 or motion == 11:
            if self.j==size:
                if self.i == size:
                    self.i-= 1
                elif self.i==0:
                    self.i+= 1
                else:
                    if randrange(0,2):
              
                        self.i += 1
                    else:
                  
                        self.i -= 1
            else:
                
                self.j+= 1
        cordinates = (self.i, self.j)
        
        return cordinates

class MavModelBasedAgent():
    def __init__(self,size=None,direction=None):
        self.grid_size = size
        self.direction = direction
        self.moves = 0
        self.puzzle_Visited = [[0 for x in range(self.grid_size)] for y in range(self.grid_size)]
        self.total_moves = 0
        self.visited_Cords = []

    def set_Direction(self,direction):
        self.direction = direction
    
    def set_Grid_Size(self,size):
        self.grid_size = size
        self.puzzle_Visited = [[0 for x in range(self.grid_size)] for y in range(self.grid_size)] 

    def add_move(self):
        self.moves += 1

    def calcualte_referance_Coordinates(self,coordinates):
        iSize , jSize = coordinates
        for i in range (0,self.grid_size):
            if (iSize >= 1):
                iSize -= 1
            if (jSize >= 1):
                jSize -= 1
    
        coordinates = iSize , jSize
        
        return coordinates
        

    def add_total_move(self):
        self.total_moves += 1

    def get_total_moves(self):
        return self.total_moves

    def puzzle_Visied(self,coordinates):
        i,j = coordinates
        self.puzzle_Visited[i][j] = 1

    def if_Puzzle_Visited(self,coordinates):
        i,j = coordinates
        if self.puzzle_Visited[i][j] == 1:
            return 1
        else:
            return 0
        
    def get_moves(self):
        return self.moves

    def get_Grid_Size(self):
        return self.grid_size
        
    def get_Direction(self):
        redo = randrange(0,12)
        return (redo)

    def get_Random_Location(self):
        return (randrange(0,self.grid_size),randrange(0,self.grid_size))
    

    def get_Location(self,cordinates):
        iSize , jSize = cordinates
        if (iSize == self.grid_size - 1 and jSize == 0 ):
            iSize == 0
        if (jSize <= self.grid_size - 2):
                jSize += 1
        else:
            if (iSize != self.grid_size - 1):
                jSize = 0
            if (iSize <= self.grid_size - 2):
                iSize += 1
        cordinates = iSize , jSize
        
        return cordinates
    
    def program(self,percept):
        _ , status = percept
        if status == 0:
            return 0
        if status == 1:
            return 1
      
        
class MavSimpleReflexAgent():
    def __init__(self,size=None,direction=None):
        self.grid_size = size
        self.direction = direction
        self.moves = 0
        self.total_moves = 0

    def set_Direction(self,direction):
        self.direction = direction
    
    def set_Grid_Size(self,size):
        self.grid_size = size

    def add_total_move(self):
        self.total_moves += 1

    def get_total_moves(self):
        return self.total_moves

    def add_move(self):
        self.moves += 1

    def get_moves(self):
        return self.moves

    def get_Grid_Size(self):
        return self.grid_size
        
    def get_Random_Direction(self):
        #print("size is ",self.grid_size)
        redo = 12
        redo = randrange(0,redo)
        #print("Direction leaving",redo)
        return (redo)

    def get_Cordinates(self,cordinates):
        #print(self.grid_size,"-------->>>",self.grid_size - 1)
        return self.direction.update_Cordinates(cordinates,self.get_Random_Direction(),self.grid_size -1)

    def get_Random_Location(self):
        return (randrange(0,self.grid_size),randrange(0,self.grid_size))

    
    def program(self,percept):
        cordinates , status = percept
        if status == 0:
            return 0
        if status == 1:
            return 1

   

class Grid(Thing):
    def create_Grid(self,n):
        self.grid = np.random.random((n,n,2,2))
        
    def get_Grid(self):
        return self.grid

    def set_Grid(self,brid):
        self.grid = copy.deepcopy(brid)
        
    def shuffle_Grid_Tiles(self,size):
        gridList = copy.deepcopy(self.grid)
        for i in range(0,size):
            for j in range(0,size):
                rots = randrange(0,17)
                rot = np.rot90(gridList[i][j],rots,(1,0))
                gridList[i][j] = copy.deepcopy(rot)
        self.grid = copy.deepcopy(gridList)

    def print_grid(self):
        print(self.grid)
     
    def rotate_clockwise(self,grid):
            grid = np.rot90(grid,1,(1,0))
            return grid

    def rotate_antiClockwise(self,grid):
            grid = np.rot90(grid)
            return grid


    

print("clear")
gridSize = int(input("Please enter grid Size "))
moves = int(input("Please enter moves "))
original_Grid = Grid()
original_Grid.create_Grid(gridSize)
puzzle_Grid = Grid()
puzzle_Grid = copy.deepcopy(original_Grid)
puzzle_Grid.shuffle_Grid_Tiles(gridSize)
myEnvi = MavEnvironment()
myEnvi.add_thing(puzzle_Grid)
myEnvi.set_Grid(puzzle_Grid)
myEnvi.set_Matching_Grid(original_Grid)
myEnvi.print_things()
myDirection = MavDirection()
myAgent = MavSimpleReflexAgent(gridSize,myDirection)
mySmartAgent = MavModelBasedAgent(gridSize)
myEnvi.add_thing(myAgent)
myEnvi.execute_action(moves)
myEnvi.total_correct_pieces()
##myEnvi.result()
myEnvi.add_thing(mySmartAgent)
myEnvi.execute_action(moves)
myEnvi.total_correct_pieces()
myEnvi.result()
