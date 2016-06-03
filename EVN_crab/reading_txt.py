i = 2
j = 3
with open('all30.txt', 'r') as f:
	text = f.read()
	text_lines = text.split('\n')

strings1 = text_lines[i-1].split()
strings2 = text_lines[j-1].split()
scan_no1 = strings1[1]
scan_no2 = strings2[1]



