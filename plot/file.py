s = 1
with open('best.log', 'w') as out_file:
    with open('new.log', 'r') as in_file:
        for line in in_file:
            out_file.write(str(s) + ' ' + line)
            s += 1
