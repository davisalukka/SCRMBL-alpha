# load text 
filename = 'test_suit.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split based on words only
import re
words = re.split(r'\W+', text)
print(words[:100])
