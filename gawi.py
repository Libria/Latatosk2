"""
=================
:mod:'gawi bawi bo game'
=================
1. base
	* 0 : gawi
	* 1 : bawi
	* 2 : bo

2. determine_winner
	Determine winner of the game.
		:param people : my hand parameter
		:param com : com's hand parameter
		:return value
		* 0 : Draw
		* 1 : player win
		* 2 : computer win

3. while (people < 0 or people > 2) :
	Check the input value ,people, is 0~2
	if people < 0 or people > 2,
	one more input
"""

import random;

def determine_winner(people, com) :
	"""
	Determine winner of the game.
	:param people : my hand parameter
	:param com : com's hand parameter
	:return value
		: 0 : Draw
		: 1 : player win
		: 2 : computer win
	"""
	if ((com == 2 and  people == 0 ) or (com == 0 and people == 2)) :
		if (com == 2 and people == 0 ) :
			win = 1
		elif (com == 0 and people == 2) :
			win = 2
	else :
		if (people > com) :
			win = 1
		elif (people < com) :
			win = 2
		elif (people == com) :
			win = 0

	return win

if __name__ == '__main__' :
	"""3. while (people < 0 or people > 2) :
		Check the input value ,people, is 0~2
		if people < 0 or people > 2,
		one more input"""
	for i in range(1,11) :	

		com = random.randint(0,2)	

		print("Show your hand (0:gawi, 1:bawi, 2:bo) ")
		people = int(input())

	"""
	Check the input value ,people, is 0~2
	if people < 0 or people > 2,
	one more input
	"""
		while (people < 0 or people > 2) :
			print("Show your Hand again (0:gawi, 1:bawi, 2:bo) ")
			people = int(input())

		win = determine_winner(people, com)	
	"""
	In determine_winner, 0 : draw, 1 : player win, 2 : computer win.
	"""
		if (win == 0) :
			print("You draw \n")
		elif (win == 1) :
			print("You win \n")
		else :
			print("You lose \n")
