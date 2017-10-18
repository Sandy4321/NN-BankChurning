import inspect
import sys
import numpy as np

def raiseNotDefined():
	file = inspect.stack()[1][1]
	line = inspect.stack()[1][2]
	method = inspect.stack()[1][3]

	print('\n\n*** Method not implemented: %s at line %s of %s\n\n' % (method, line, file))
	sys.exit(1)

def raiseError(message):
	file = inspect.stack()[1][1]
	line = inspect.stack()[1][2]
	method = inspect.stack()[1][3]

	print('\n\n*** Error %s at line %s of %s: %s\n\n' % (method, line, file, message))
	sys.exit(1)
