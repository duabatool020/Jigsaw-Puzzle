# Jigsaw-Puzzle
# Jigsaw-Puzzle
This program will solve the problem given below:<br>
Consider you have an agent for a Jigsaw puzzle. The goal of the agent is to complete the Jigsaw 
puzzle in fixed number of moves. Your implementation should be modular so that the sensors, 
actuators, and environment characteristics (size etc.) can be changed easily. You need to 
implement a graph based search agent with visited state check.<br>
You are required to formulate this problem by defining the four parameters:<br> 
1. Initial State <br>
2. Successor Function <br>
3. Goal Test <br>
4. Path cost <br>
Problem Description: <br>
Available actions are Left, Right, Up, Down, NoOp, Rotate_left and Rotate_right. <br>
Rotate_left rotates the cell content by 90-degree counter clockwise<br>
Rotate_right rotates the cell content by 90-degree clockwise <br>
The termination criterion is that the agent runs out of moves or the puzzle is completed. 
Environment announces when this criteria is met and stops the execution. <br>
The performance of an agent is calculated after the termination criteria is met. The 
performance measure of an agent is the (# of correctly placed items) / (number of steps 
used). Note that there will be early termination if puzzle is solved before expiry of 
maximum number of moves. <br>
• The environment is deterministic and partially observable. <br>
• Agent knows the size of puzzle (grid n x n) and the content of the cell they land in, location 
of the landing cell (coordinates) is not known. <br>
•The perception is given by the environment and includes, cell coordinates and if the current 
piece in the cell is rightly placed or not. <br>
• Starting position of the agent is random and not known beforehand plus puzzle contents 
are randomized at each start.<br>
